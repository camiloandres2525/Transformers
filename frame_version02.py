from fastapi import FastAPI
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from pdfminer.high_level import extract_text_to_fp
from io import StringIO

app = FastAPI()


with open('CONTRATO.pdf', 'rb') as archivo_pdf:
    
    salida_texto = StringIO()
    extract_text_to_fp(archivo_pdf, salida_texto)
    contenido = salida_texto.getvalue()

the_model = 'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'
tokenizer = AutoTokenizer.from_pretrained(the_model)
model = AutoModelForQuestionAnswering.from_pretrained(the_model)

def pregunta_respuesta(contexto, pregunta):
    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
    salida = nlp({'question': pregunta, 'context': contexto})
    
    return salida['answer']

@app.get("/")
def read_root():
    return {"message": "API funcionando."}

@app.post("/preguntas")
def obtener_respuesta(pregunta: str):
    respuesta = pregunta_respuesta(contenido, pregunta)
    return {"respuesta": respuesta}
