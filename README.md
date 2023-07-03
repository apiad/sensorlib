# Instrucciones

Instalar las dependencias:

```
pip install -r requirements.txt
```

Descargar el modelo de `spacy` necesario:

```
spacy download en_core_web_lg
```

Ejecutar la aplicación:

```
streamlit run app.py
```

Visitar <localhost:8501> para ver el demo.

![](screenshot.png)

- En `experiments.py` están los códigos de experimentación.
- En `sensorlib.py` están refactorizados los métodos y funcionalidades necesarias.
- El código en `app.py` puede servir para entender como usar los métodos en `sensorlib.py`.

**NOTA**: La primera vez que se ejecuta el código se debe descargar un modelo de `transformers` de 2GB aproximadamente. Esto puede demorar.
