# Sensor de NER para dominio general

Este sensor está implementado usando modelos de lenguaje y
técnicas de prompt-engineering (k-shot prompting) para permitir
su uso sin necesidad de entrenamiento o fine-tuning.

Para ello es necesario acceder a un modelo de embeddings (BERT o superior) y un modelo de LLM (Gemma-3b o superior) que pueden ser
utilizados desde un servicio o montados en una API local usando LM Studio <lmstudio.ai> o cualquier otro servidor de LLM local que provea
una API compatible con OpenAI (todos lo hacen).

La calidad de los resultados es muy dependiente del modelo LLM usado,
con modelos más grandes se obtiene mejor precisión.

Como primer paso se indexan y embeben todos los ejemplos entrenantes
que servirán posteriormente para el k-shot.
Luego se genera un prompt por cada ejemplo a extraer y se consulta al LLM.
Finalmente se parsea la respuesta y se convierte a formato BRAT.

Es muy importante ejecutar los modelos en GPU pues de lo contrario
la respuesta es muy lenta.
LM Studio provee soporte para correr los modelos total o parcialmente
en GPU con facilidad.

## Instrucciones generales

Instalar las dependencias:

```
pip install -r requirements.txt
```

## Para ejecutar el demo

Configurar `.streamlit/secrets.toml`:

```
[openai]

url = "https://api.mistral.ai/v1/"
key = "WiT9wnodUHLECpONpdRLtWo95ETu8MxU"
embedding_model = "mistral-embed"
llm_model = "open-mixtral-8x7b"
```

> **NOTA:** Esta es una configuración de ejemplo que funciona con nuestro token de <mistral.ai>. Se puede usar para probar el demo, pero no para producción, pues tiene límite de gasto.

Ejecutar la aplicación:

```
streamlit run app.py
```

> **NOTA:** El código en `app.py` puede servir para entender como usar los métodos en `sensorlib.py`.
> En `sensorlib.py` están refactorizados los métodos y funcionalidades necesarias.

Visitar <localhost:8501> para ver el demo de reconocimiento de entidades usando LLMs.

## Ejecución con modelos locales

- Instalar LM Studio de <lmstudio.ai>.
- Descargar desde LM Studio un modelo de LLM suficientemente bueno (hemos probado con `Gemma-3b` y `Llama3-7b`.)
- Descargar desde LM Studio un modelo de embedding (cualquiera funciona.)
- Activar la API de LM Studio (se puede hacer por CLI o en la UI).

Para usar la aplicación demo, se debe configurar en `.streamlit/secrets.toml` con la URL y modelos locales.

Para usar la biblioteca `sensorlib.py` directamente, solamente es necesario pasar una instancia de `OpenAI` correctamente configurada.

## Workflow completo

En la biblioteca `sensorlib.py` está todo el código refactorizado para ser usado. Se separa en varias partes la funcionalidad para permitir debuggear y almacenar los resultados intermedios.

El listado de métodos a ejecutar en orden es el siguiente:

1. `build_taxonomy`: Procesar el archivo `CATEGORIAS.txt` correposdiente y obtener un diccionario en formato conveniente. Almacene esto mientras no cambie la taxonomía.
2. `parse_examples`: Procesar los ejemplos entrenantes y obtener una reprsentación en forma de diccionario. Guarde esto mientras no cambie el training set.
3. `embed`: Computar los embeddings de los ejemplos entrenantes. Guarde esto mientras no cambie el training set.

Estos pasos son de preprocesamiento y pueden demorar de 5 a 10 minutos en función del modelo de embedding usado y el tamaño del training set, pero se pueden cachear.

Luego, para el proceso de extracción de un nuevo texto.

1. `get_k_shot`: Obtener los ejemplos entrenantes más similares para el prompt.
2. `build_prompt`: Construir el prompt para pasar al LLM.
3. `reply`: Llamar al LLM y obtener la respuesta.
4. `convert_to_ann`: Convertir la respuesta a formato ANN.

En función de la calidad del LLM usado, la respuesta en `reply` puede contener errores. En caso de no devolver un JSON correcto, generalmente funciona volver a ejecutar el `reply`. Esto no ocurre con modelos del estado del arte, e.g., `mixtral-8x7b` pero si puede ocurrir con modelos más pequeños.

## Notas finales

Este enfoque se escogió porque es adaptable fácilmente a cualquier taxonomía y dominio sin necesidad de re-entrenar y no requiere de un training set muy grande. Con pocos cientos de ejemplos bien variados es suficiente.

Sin embargo, si se necesita una precisión mucho mayor, o si los modelos LLM escogidos no son muy potentes, será conveniente hacer fine-tuning. Llegado este punto podemos verlo.

Por otro lado, si la infraestructura del cliente no permite ejecutar un LLM suficientemente bueno, podemos probar un enfoque solo basado en embeddings, pero necesariamente menor preciso. Igualmente, llegado este punto podemos analizarlo.
