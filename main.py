import tempfile
import os
from io import BytesIO
from flask import Flask, request, redirect, send_file
from skimage import io
from skimage.transform import resize
from tensorflow.keras.models import load_model
import shutil
import base64
import glob
import numpy as np

app = Flask(__name__)

ruta_carpeta_figuras = "figuras"

main_html = """
<html>
<head>
<script src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
<script>
  var mousePressed = false;
  var lastX, lastY;
  var ctx;

  function getRndInteger(min, max) {
    return Math.floor(Math.random() * (max - min)) + min;
  }

  function InitThis() {
    ctx = document.getElementById('myCanvas').getContext("2d");

    $('#myCanvas').mousedown(function (e) {
      mousePressed = true;
      Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false);
    });

    $('#myCanvas').mousemove(function (e) {
      if (mousePressed) {
        Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true);
      }
    });

    $('#myCanvas').mouseup(function (e) {
      mousePressed = false;
    });
    $('#myCanvas').mouseleave(function (e) {
      mousePressed = false;
    });
  }

  function Draw(x, y, isDown) {
    if (isDown) {
      ctx.beginPath();
      ctx.strokeStyle = 'black';
      ctx.lineWidth = 11;
      ctx.lineJoin = "round";
      ctx.moveTo(lastX, lastY);
      ctx.lineTo(x, y);
      ctx.closePath();
      ctx.stroke();
    }
    lastX = x; lastY = y;
  }

  function clearArea() {
    // Use the identity matrix while clearing the canvas
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  }

  //https://www.askingbox.com/tutorial/send-html5-canvas-as-image-to-server
  function prepareImg() {
    var canvas = document.getElementById('myCanvas');
    document.getElementById('myImage').value = canvas.toDataURL();
  }
</script>
</head>
<body onload="InitThis();">
  <div align="center">
    <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/PSX-DualShock-Controller.jpg/375px-PSX-DualShock-Controller.jpg" width="300"/>
  </div>
  <div align="center">
    <h1 id="mensaje"></h1>
    <canvas id="myCanvas" width="200" height="200" style="border:2px solid black"></canvas>
    <br/>
    <br/>
    <button onclick="javascript:clearArea();return false;">Borrar</button>
  </div>
  <div align="center">
    <form method="post" action="upload" onsubmit="javascript:prepareImg();"  enctype="multipart/form-data">
      <input id="myImage" name="myImage" type="hidden" value="">
      <input id="bt_upload" type="submit" value="Enviar">
    </form>
  </div>
  <script>
    $(document).ready(function(){
      var mensaje = "{{ mensaje }}";
      if (mensaje !== "") {
        $("#mensaje").text(mensaje);
      }
    });
  </script>
</body>
</html>
"""

@app.route("/")
def main():
    mensaje = request.args.get('mensaje', '')  # Obtener el mensaje de la URL
    return main_html.replace('{{ mensaje }}', mensaje)

@app.route('/upload', methods=['POST'])
def prepare_and_save():
    try:
        # check if the post request has the file part
        img_data = request.form.get('myImage').replace("data:image/png;base64,","")
        with tempfile.NamedTemporaryFile(delete = False, mode = "w+b", suffix='.png',dir=ruta_carpeta_figuras) as fh:
            fh.write(base64.b64decode(img_data))
        
        prepare_dataset()

        # Elimina todo el contenido de la carpeta "numeros"
        shutil.rmtree(ruta_carpeta_figuras)
        # Vuelve a crear la carpeta "numeros" vacía
        os.makedirs(ruta_carpeta_figuras)

        valor_predicho = predecir()

        return redirect("/?mensaje=" + str(valor_predicho), code=302)
    except Exception as err:
        print("Error occurred")
        print(err)


    return redirect("/", code=302)

def prepare_dataset():
  images = []
  filelist = glob.glob('{}/*.png'.format("figuras"))
  images_read = io.concatenate_images(io.imread_collection(filelist))
  images_read = images_read[:, :, :, 3]
  images.append(images_read)
  images = np.vstack(images)
  np.save('X.npy', images)


def predecir():
  X_raw = np.load('X.npy')
  X_raw = X_raw/255.
  X = []
  size = (28,28)
  for x in X_raw:
      X.append(resize(x, size))
  X = np.array(X)
  X = X[...,None]
  im = X[0]
  
  model = load_model("./modelo_entrenado_figuras.h5")
  prediccion = model.predict(im[None,:,:,:])[0]
  digit_number = {1: "X", 2:"Cuadradro", 3:"Triangulo", 4: "O"}

  return digit_number[prediccion.argmax()]



if __name__ == "__main__":
    if not os.path.exists(str(ruta_carpeta_figuras)):
        os.mkdir(str(ruta_carpeta_figuras))
    app.run()
