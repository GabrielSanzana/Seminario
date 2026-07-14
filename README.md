# Configuración de Google Earth Engine

Antes de ejecutar el notebook es necesario habilitar la **Google Earth Engine API** en su proyecto de Google Cloud.

## 1. Crear un proyecto en Google Cloud

Si aún no dispone de un proyecto:

1. Ingrese a https://console.cloud.google.com/
2. Cree un nuevo proyecto.
3. Guarde el **ID del proyecto**, ya que será utilizado posteriormente.

---

## 2. Habilitar la Google Earth Engine API

1. Ingrese a **Google Cloud Console**.
2. Diríjase a **APIs y servicios**.
3. Seleccione **Habilitar APIs y servicios**.
4. Busque **Google Earth Engine API**.
5. Seleccione la API y presione **Habilitar**.

---

## 3. Modificar el proyecto en el código

Antes de ejecutar el notebook, cambie la siguiente línea:

```python
PROJECT_ID = "id_proyecto"
```

por el ID de su proyecto.

Ejemplo:

```python
PROJECT_ID = "ee-patricio21"
```

---

## 4. Ejecutar el notebook

El proyecto fue desarrollado para ejecutarse directamente en **Google Colab**.

Una vez configurado el `PROJECT_ID`, solo debe ejecutar las celdas del notebook en orden.

Durante la primera ejecución Google solicitará autorización para acceder a Earth Engine. Inicie sesión con la cuenta que tiene acceso al proyecto y acepte los permisos.
