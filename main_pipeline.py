from typing import Annotated, List, Union, Optional
from pydantic import BaseModel, ConfigDict, SkipValidation
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams, PointStruct, ErrorResponse
from pydantic_settings import SettingsConfigDict, BaseSettings
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
import utils.qwen_qa_utils as custom_qwen
import json
import time


import os
from typing import BinaryIO, List, Any, Union
import io
from minio import Minio
from minio.error import S3Error
import json
import ast
import base64

from MMLongDocEval.extract_answer import extract_answer
from MMLongDocEval.eval_score import eval_score, eval_acc_and_f1, show_results



def read_json(filename):
    with open(filename) as json_data:
        data = json.load(json_data)
        json_data.close()
    return data


def read_jsonl(filename):
    with open(filename) as f:
        data = [json.loads(line) for line in f]
        f.close()
    return data




class Message(BaseModel):
    model_config = ConfigDict(extra='ignore')

    type: str
    text: Union[str, None] = None
    image: Union[str, None] = None
    image_url: Union[str, None] = None

class EmbedRequest(BaseModel):
    messages: List[Message]

class MessageEmbedding(BaseModel):
    message_id: int
    embedding: List[float]

class EmbedResponse(BaseModel):
    messages: List[MessageEmbedding]

class EmbedErrorResponse(BaseModel):
    detail: str
    

def extract_answer_qwen_api(question, output, prompt):
    tt = custom_qwen.ModelMessageDict()
    tt.add_text_content(prompt)
    answer=f"Question: {question}\nAnalysis:{output}"
    tt.add_text_content(answer)
    result = custom_qwen.send_messasge(messages=[tt], base_url='http://192.168.19.127:8888/v1')
    return result[1][0]



# ========== Client ==========
class EmbeddingClient:
    """
    A simple client for the embedding service.
    """

    def __init__(self, base_url: str, timeout: int = 30):
        """
        Args:
            base_url: The base URL of the embedding service (e.g. http://localhost:8000).
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout


    def get_text_embedding(self, text: str) -> EmbedResponse:

        message = Message(type="text", text=text)
        request_payload = EmbedRequest(messages=[message]).model_dump()  # or .dict() for older Pydantic
        url = f"{self.base_url}/embed"  # adjust the endpoint path as needed
        response = requests.post(url, json=request_payload, timeout=self.timeout)
        response.raise_for_status()  # raise exception for HTTP errors
        try:
            return EmbedResponse.model_validate(response.json())  # or .parse_obj() for older Pydantic
        except Exception as e:
            raise ValueError(f"Invalid response format: {e}") from e


    def get_embeddings(self, messages: List[Message]) -> Union[EmbedResponse, EmbedErrorResponse]:
        """
        Send a list of messages to the service and retrieve the embedding.

        Args:
            messages: A list of Message objects.

        Returns:
            EmbedResponse containing the message_id and embedding vector.

        Raises:
            requests.RequestException: If the HTTP request fails.
            ValueError: If the response cannot be parsed into EmbedResponse.
        """
        # Build the request payload using the EmbedRequest model
        print(messages)
        request_payload = EmbedRequest(messages=messages).model_dump()  # or .dict() for older Pydantic

        # Send POST request
        url = f"{self.base_url}/embed"  # adjust the endpoint path as needed
        response = requests.post(url, json=request_payload, timeout=self.timeout)
        response.raise_for_status()  # raise exception for HTTP errors

        # Parse and validate the response
        try:
            return EmbedResponse.model_validate(response.json())  # or .parse_obj() for older Pydantic
        except Exception as e:
            raise ValueError(f"Invalid response format: {e}") from e


    
    
    
# ========== Main pipeline ==========
if __name__ == "__main__":
    
    retrieve_prompt="Answer the question based on given context. Elements of context are presented after the question. Each piece of context starts with "
    HNKdict=read_json(f"{os.getcwd()}/file_hash_comparison.json")       #file comparing filenames to hashcodes
    filename=f"{os.getcwd()}/MMLongDoc_answers.json"
    if(os.path.exists(filename)):    
        dataset=read_json(filename)
    else:
        dataset=read_json('/home/user/RAG/MMLongBench_Doc/data/samples.json')   #path to samples from original MMLongBench_Doc
    reserve_check=True
    with open(f"{os.getcwd()}/MMLongDocEval/prompt_for_answer_extraction.md",'r') as f:
        extract_prompt = f.read()
        
    mc=Minio(
                endpoint='localhost:9000',
                access_key='minio',
                secret_key='minio123',
                secure=False
            )
            
    bucket_name='pdf-processing'
    limit=20
    
    for k, caser in enumerate(dataset):
        start_time=time.time()
        model_answer_time=0
        model_extract_time=0


        #костыль для частичной обработки файлов
        if(k<-1):
            continue

        caser['status']='unknown'
        
        if('response' in caser.keys()):
            print(f"\n>>>> Already processed! {k}/{len(dataset)}\n\n")
            continue
        
        print(f"\n>>> question {k}/{len(dataset)}\n\n")
        
        query=caser['question']
        if(caser['doc_id'] not in HNKdict.keys()):
            print(f"doc {caser['doc_id']} not in base")
            caser['response']='failed to find the document'
            continue
        else:
            fileHash=HNKdict[caser['doc_id']]
            print(fileHash)
        messages = [
            Message(type="text", text=query)
        ]
    
        # Initialize the client (point to your actual service URL)
        clientQwenEmb = EmbeddingClient(base_url="http://192.168.19.127:10115/embedding")
        clientQdrant=QdrantClient(host="localhost", port=6333)
        clientQwenMsg=custom_qwen.ModelMessageDict()
        embeds=None
        search_result=None
        model_answer=None
        retrieves=None
        score=0
        extracted_res=None
        try:
            result = clientQwenEmb.get_embeddings(messages)
            #print(f"Message ID: {result.message_id}")
            #print(f"Embedding (first 5 values): {result.embedding[:5]}...")
            embeds=result.messages[0].embedding
            #print(f"Total Result: {result}")
        except Exception as e:
            print(f"Error: {e}")
            
        if(embeds!=None):
            search_result = clientQdrant.query_points(
                                    collection_name="documents_MMLongDoc",
                                    query=embeds,
                                    with_payload=True,
                                    query_filter=Filter(must=[FieldCondition(key="file_hash", match=MatchValue(value=fileHash))]),
                                    limit=limit
                                ).points
            print(f"\n\nQuery: {query}\n\n")
            
            for elem in search_result:
                print(f"{elem.payload['text']} <-> {elem.score}\n\n")

            print(len(search_result))
        else:
            print('embeds are None')

            
        #search_result=None
        
        
        if(search_result!=None and len(search_result)>0):
            prompt=''
            retrieves=[]
            tt = custom_qwen.ModelMessageDict()
            tt.add_text_content(f"Question: {query}")
            content=None
            for elem in search_result:
                if(elem.payload['element_type'] != 'image'):
                    tt.add_text_content(elem.payload['text'])
                    content=elem.payload['text']
                if(elem.payload['element_type'] == 'image'):
                    #try:
                        response=mc.get_object(bucket_name, f"{elem.payload['img_path']}")
                        encoded_string = base64.b64encode(response.data).decode('utf-8')
                        response.close()
                        response.release_conn()
                        tt.add_img_content_base64(encoded_string)
                        content=elem.payload['img_path']
                    #except:
                        #continue
                retrieves.append({'qdrant_id': elem.id,'type':elem.payload['element_type'], 'file_hash':elem.payload['file_hash'], 'content': elem.payload['text'],'page_idx': elem.payload['original_element']['page_idx'], 'score':elem.score})
            try:
                result = custom_qwen.send_messasge(messages=[tt], base_url='http://192.168.19.127:8888/v1')
            except:
                caser['status']='failed to connect to LLM'
                print(f"failed to connect to model")
                continue
            model_answer_time=time.time()-start_time
            start_time=time.time()
            if(str(result[1])=='None'):
                break
            model_answer=result[1][0]
            extracted_res=extract_answer_qwen_api(query, model_answer, extract_prompt)
            model_extract_time=time.time()-start_time
            print(f">>> Extracted answer:\n{extracted_res}\n\n>>> Correct answer: {caser['answer']}\n\n\n")
            caser['response'] = model_answer
            caser['extracted_res'] = extracted_res
            caser['score'] = score
            caser['retrieves'] = retrieves
            caser['status'] = 'compleated'
        else:
            print('retrieves are None')
            caser['status']='no retrieves'
            
        print(f"\n answering time: {model_answer_time}; answer extraction time: {model_extract_time} \n")
        with open('proceeding_time.txt', 'a') as timesF:
            timesF.write(f"{model_answer_time};{model_extract_time}\n")
        

        
        if(reserve_check):
            with open('MMLongDoc_answers.json', 'w') as f:
                json.dump(dataset, f)
        
    with open('MMLongDoc_answers.json', 'w') as f:
        json.dump(dataset, f)
        
    
