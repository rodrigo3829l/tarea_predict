from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd
import logging

app = Flask(__name__)
# Configurar el registro
logging.basicConfig(level=logging.DEBUG)

model = joblib.load('modelo.pkl')
app.logger.debug('Modelo cargado correctamente.')

scaler = joblib.load('scaler.pkl')
app.logger.debug('Escaler cargado correctamente.')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obtener los datos enviados en el request
        pay_0 = int(request.form['pago'])

        # Crear un DataFrame con los datos
        data_df = pd.DataFrame([[pay_0]], columns=['PAY_0'])
        df_scaled = scaler.transform(data_df)
        df = pd.DataFrame(df_scaled, columns=['PAY_0'])
        app.logger.debug(f'DataFrame creado: {df}')
        
        # Realizar predicciones
        prediction = model.predict(df)
        app.logger.debug(f'Predicción: {prediction[0]}')
        data = 'No se retrasará' if prediction[0] == 0 else 'Propenso a retrasarse'
        # Devolver las predicciones como respuesta JSON
        return jsonify({'retraso': data})
    except Exception as e:
        app.logger.error(f'Error en la predicción: {str(e)}')
        return jsonify({'error': str(e)}), 400
if __name__ == '__main__':
    app.run(debug=True)
