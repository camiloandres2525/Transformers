from textwrap import wrap
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from pdfminer.high_level import extract_text_to_fp
from io import StringIO

# Abre el archivo PDF en modo de lectura binaria
with open('Brochure insualiados PDF.pdf', 'rb') as archivo_pdf:
    salida_texto = StringIO()
    
    extract_text_to_fp(archivo_pdf, salida_texto)
    contenido = salida_texto.getvalue()
    print(contenido)

the_model = 'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'
tokenizer = AutoTokenizer.from_pretrained(the_model)
model = AutoModelForQuestionAnswering.from_pretrained(the_model)
print(model)

# tokenizacion
contexto = 'soy chatbot'
pregunta = 'quien eres?'

encode = tokenizer.encode_plus(pregunta, contexto, return_tensors='pt')
input_ids = encode['input_ids'].tolist()
tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
for id, token in zip(input_ids[0], tokens):
    print('{:<12} {:>16}'.format(token, id))
    print(' ')
    
# inferencia (pregunta - respuesta)
nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)
salida = nlp({'question': pregunta, 'context': contexto})
print(salida)

def pregunta_respuesta(model,contexto,nlp):
    #imprimir context
    #print('Contexto')
    print('Hola, ingresa la pregunta de acuerdo a la informacion suministrada: ')
    print('--------------------')
    #print('\n'.join(wrap(contexto)))
    
    continuar = True
    while continuar:
        print('\nPregunta: ')
        print('--------------------')
        pregunta = str(input())
        continuar = pregunta != 'salir'

        if continuar:
            salida = nlp({'question': pregunta, 'context': contexto})
            print('\nRespuesta:')
            print('--------------------')
            print(salida['answer'])
            
contexto = contenido
pregunta_respuesta(model, contexto, nlp)

