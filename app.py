from flask import Flask, render_template, jsonify, send_file
import pandas as pd
from sklearn.linear_model import LinearRegression
from sqlalchemy import create_engine
from flask import Flask, send_from_directory
from flask import request
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures


app = Flask(__name__)

# Conexión a SQL Server
def get_db_connection():
    server = 'DESKTOP-DSO1UAU\\SQLEXPRESS01'  # Reemplaza con tu servidor de SQL Server
    database = 'dash_filtro'  # Base de datos
    username = ''  # Reemplaza con tu usuario
    password = ''  # Reemplaza con tu contraseña
    driver = 'ODBC Driver 17 for SQL Server'  # Asegúrate de tener el driver instalado
    connection_string = f'mssql+pyodbc://{username}:{password}@{server}/{database}?driver={driver}'
    try:
        engine = create_engine(connection_string)
        return engine.connect()
    except Exception as e:
        print(f"Error al conectar a la base de datos: {e}")
        return None
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/kpis')
def kpis():
    valor_kpi1 = 8
    valor_kpi2 = 8
    valor_kpi3 = 8
    return render_template('kpis.html', valor_kpi1=valor_kpi1, valor_kpi2=valor_kpi2, valor_kpi3=valor_kpi3)

@app.route('/img/<filename>')
def send_image(filename):
    image_path = f'C:\\dash_pro\\img\\{filename}'
    return send_file(image_path, mimetype='image/png')


@app.route('/regression_data')
def regression_data():
    conn = get_db_connection()
    if conn is None:
        return jsonify(error="Error al conectar con la base de datos"), 500
    
    sql_query = 'SELECT * FROM pred_hum'
    try:
        data = pd.read_sql(sql_query, conn)
    except Exception as e:
        print(f"Error al realizar la consulta SQL: {e}")
        return jsonify(error="Error al realizar la consulta SQL"), 500
    #finally:
        #conn.close()

    y = data['humedad_f1']  # Variable dependiente
    X = data.drop(columns=['humedad_f1', 'fecha'])  # Variables independientes

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    actual_vs_predicted = list(zip(y, y_pred))

    r2_score = model.score(X, y)
    
    coefficients = model.coef_
    intercept = model.intercept_
    equation = f"y = {intercept:.4f}"
    for i, coef in enumerate(coefficients):
        equation += f" + ({coef:.4f})*x{i+1}"

    return jsonify(actual_vs_predicted=actual_vs_predicted, r2_score=r2_score, equation=equation)





@app.route('/regression_data2')
def regression_data2():
    conn = get_db_connection()
    if conn is None:
        return jsonify(error="Error al conectar con la base de datos"), 500
    
    sql_query = 'SELECT * FROM pred_hum'
    try:
        data = pd.read_sql(sql_query, conn)
    except Exception as e:
        print(f"Error al realizar la consulta SQL: {e}")
        return jsonify(error="Error al realizar la consulta SQL"), 500
    #finally:
        #conn.close()

    y = data['humedad_f1']  # Variable dependiente
    X = data.drop(columns=['humedad_f1', 'fecha'])  # Variables independientes

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    actual_vs_predicted = list(zip(y, y_pred))

    r2_score = model.score(X, y)
    
    coefficients = model.coef_
    intercept = model.intercept_
    equation = f"y = {intercept:.4f}"
    for i, coef in enumerate(coefficients):
        equation += f" + ({coef:.4f})*x{i+1}"

    return jsonify(actual_vs_predicted=actual_vs_predicted, r2_score=r2_score, equation=equation)





@app.route('/regression_data3')
def regression_data3():
    conn = get_db_connection()
    if conn is None:
        return jsonify(error="Error al conectar con la base de datos"), 500
    
    sql_query = 'SELECT * FROM pred_hum'
    try:
        data = pd.read_sql(sql_query, conn)
    except Exception as e:
        print(f"Error al realizar la consulta SQL: {e}")
        return jsonify(error="Error al realizar la consulta SQL"), 500
    #finally:
        #conn.close()

    y = data['humedad_f1']  # Variable dependiente
    X = data.drop(columns=['humedad_f1', 'fecha'])  # Variables independientes

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    actual_vs_predicted = list(zip(y, y_pred))

    r2_score = model.score(X, y)
    
    coefficients = model.coef_
    intercept = model.intercept_
    equation = f"y = {intercept:.4f}"
    for i, coef in enumerate(coefficients):
        equation += f" + ({coef:.4f})*x{i+1}"

    return jsonify(actual_vs_predicted=actual_vs_predicted, r2_score=r2_score, equation=equation)

@app.route('/resumen')
def resumen():
    # No es necesario pasar datos aquí si se van a cargar a través de AJAX
    return render_template('resumen.html')

@app.route('/last_24_records_data')
def last_24_records_data():
    conn = get_db_connection()
    if conn is None:
        return jsonify(error="Error al conectar con la base de datos"), 500

    sql_query = '''
    SELECT TOP 24 fecha, p_keke_f1, p_keke_f2, p_keke_f3
    FROM filtro_cu
    ORDER BY fecha DESC
    '''
    try:
        data = pd.read_sql(sql_query, conn)
        data = data.sort_values('fecha')  # Aseguramos que las fechas estén en orden ascendente
        # Limpieza: reemplazar NaN por None
        data = data.where(pd.notnull(data), None)
        result = data.to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        print(f"Error al realizar la consulta SQL: {e}")
        return jsonify(error="Error al realizar la consulta SQL"), 500
    finally:
        conn.close()


@app.route('/last_24_cycle_data')
def last_24_cycle_data():
    conn = get_db_connection()
    if conn is None:
        return jsonify(error="Error al conectar con la base de datos"), 500

    sql_query = '''
    SELECT TOP 24 fecha, n_cic_f1, n_cic_f2, n_cic_f3
    FROM filtro_cu
    ORDER BY fecha DESC
    '''
    try:
        data = pd.read_sql(sql_query, conn)
        data = data.sort_values('fecha')  # Aseguramos que las fechas estén en orden ascendente
        # Limpieza: reemplazar NaN por None
        data = data.where(pd.notnull(data), None)
        result = data.to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        print(f"Error al realizar la consulta SQL: {e}")
        return jsonify(error="Error al realizar la consulta SQL"), 500
    finally:
        conn.close()


@app.route('/average_cycle_time_data')
def average_cycle_time_data():
    conn = get_db_connection()
    if conn is None:
        return jsonify(error="Error al conectar con la base de datos"), 500
    
    sql_query = '''
    SELECT TOP 24 fecha, prom_cic_f1, prom_cic_f2, prom_cic_f3
    FROM filtro_cu
    ORDER BY fecha DESC
    '''

    try:
        data = pd.read_sql(sql_query, conn)
        # Limpieza: reemplazar NaN por None
        data = data.where(pd.notnull(data), None)
        result = data.to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        print(f"Error al realizar la consulta SQL: {e}")
        return jsonify(error="Error al realizar la consulta SQL"), 500
    finally:
        conn.close()

@app.route('/drawCharts_niveltk')
def drawCharts_niveltk():
    conn = get_db_connection()
    if conn is None:
        return jsonify(error="Error al conectar con la base de datos"), 500
    
    sql_query = '''
    SELECT TOP 24 fecha, lit_tk_001, lit_tk_002
    FROM filtro_cu
    ORDER BY fecha DESC
    '''

    try:
        data = pd.read_sql(sql_query, conn)
        # Limpieza: reemplazar NaN por None
        data = data.where(pd.notnull(data), None)
        result = data.to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        print(f"Error al realizar la consulta SQL: {e}")
        return jsonify(error="Error al realizar la consulta SQL"), 500
    finally:
        conn.close()


@app.route('/drawCharts_t_fil')
def drawCharts_t_fil():
    conn = get_db_connection()
    if conn is None:
        return jsonify(error="Error al conectar con la base de datos"), 500
    
    sql_query = '''
    SELECT TOP 24 fecha, t_filt_f1, t_filt_f2 ,t_filt_f3
    FROM filtro_cu
    ORDER BY fecha DESC
    '''

    try:
        data = pd.read_sql(sql_query, conn)
        # Limpieza: reemplazar NaN por None
        data = data.where(pd.notnull(data), None)
        result = data.to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        print(f"Error al realizar la consulta SQL: {e}")
        return jsonify(error="Error al realizar la consulta SQL"), 500
    finally:
        conn.close()

@app.route('/predict_queque_weight')
def predict_queque_weight():
    conn = get_db_connection()
    if conn is None:
        return jsonify(error="Error al conectar con la base de datos"), 500
    
    sql_query = 'SELECT * FROM filtro_cu'  # Asegúrate de cambiar 'tu_tabla_datos' por el nombre real de tu tabla
    try:
        data = pd.read_sql(sql_query, conn)
        data.fillna(0, inplace=True)  # Reemplaza NaN por 0
        y = data['p_keke_f1']  # Cambia 'p_keke_f1' por tu variable objetivo
        X = data[['n_cic_f1', 'prom_cic_f1', 'lit_tk_001', 'sol_cu', 't_llen_f1', 't_filt_f1', 't_pren_f1', 't_sec_f1']]  # Reemplaza con las columnas correctas
      

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)
        prediction = model.predict(X_poly)

        r2_value = r2_score(y, prediction)

        return jsonify(actual=y.tolist(), predicted=prediction.tolist(), r2_value=r2_value)

         
    except Exception as e:
        print(f"Error al realizar la consulta SQL o la predicción: {e}")
        return jsonify(error="Error al realizar la consulta SQL o la predicción"), 500
    finally:
        conn.close()


@app.route('/predict_queque_weight2')
def predict_queque_weight2():
    conn = get_db_connection()
    if conn is None:
        return jsonify(error="Error al conectar con la base de datos"), 500
    
    sql_query = 'SELECT * FROM filtro_cu'  # Asegúrate de cambiar 'tu_tabla_datos' por el nombre real de tu tabla
    try:
        data = pd.read_sql(sql_query, conn)
        data.fillna(0, inplace=True)  # Reemplaza NaN por 0
        y = data['p_keke_f2']  # Cambia 'p_keke_f1' por tu variable objetivo
        X = data[['n_cic_f2', 'prom_cic_f2', 'lit_tk_001', 'sol_cu', 't_llen_f2', 't_filt_f2', 't_pren_f2', 't_sec_f2']]  # Reemplaza con las columnas correctas
      

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)
        prediction = model.predict(X_poly)

        r2_value = r2_score(y, prediction)

        return jsonify(actual=y.tolist(), predicted=prediction.tolist(), r2_value=r2_value)

         
    except Exception as e:
        print(f"Error al realizar la consulta SQL o la predicción: {e}")
        return jsonify(error="Error al realizar la consulta SQL o la predicción"), 500
    finally:
        conn.close()


@app.route('/predict_queque_weight3')
def predict_queque_weight3():
    conn = get_db_connection()
    if conn is None:
        return jsonify(error="Error al conectar con la base de datos"), 500
    
    sql_query = 'SELECT * FROM filtro_cu'  # Asegúrate de cambiar 'tu_tabla_datos' por el nombre real de tu tabla
    try:
        data = pd.read_sql(sql_query, conn)
        data.fillna(0, inplace=True)  # Reemplaza NaN por 0
        y = data['p_keke_f3']  # Cambia 'p_keke_f1' por tu variable objetivo
        X = data[['n_cic_f3', 'prom_cic_f3', 'lit_tk_001', 'sol_cu', 't_llen_f3', 't_filt_f3', 't_pren_f3', 't_sec_f3']]  # Reemplaza con las columnas correctas
      

        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)
        prediction = model.predict(X_poly)

        r2_value = r2_score(y, prediction)

        return jsonify(actual=y.tolist(), predicted=prediction.tolist(), r2_value=r2_value)

         
    except Exception as e:
        print(f"Error al realizar la consulta SQL o la predicción: {e}")
        return jsonify(error="Error al realizar la consulta SQL o la predicción"), 500
    finally:
        conn.close()


@app.route('/latest_humidity_data')
def latest_humidity_data():
    conn = get_db_connection()
    if conn is None:
        return jsonify(error="Error al conectar con la base de datos"), 500

    sql_query = '''
    SELECT TOP 1 humedad_f1
    FROM pred_hum
    ORDER BY fecha DESC
    '''

    try:
        data = pd.read_sql(sql_query, conn)
        if not data.empty:
            latest_humidity = data.loc[0, 'humedad_f1']
            print(f"Último valor de humedad: {latest_humidity}")
            return jsonify(humidity=latest_humidity)
        else:
            print("No se encontraron datos de humedad en la base de datos.")
            return jsonify(error="No se encontraron datos de humedad"), 404
    except Exception as e:
        print(f"Error al realizar la consulta SQL: {e}")
        return jsonify(error="Error al realizar la consulta SQL"), 500
    finally:
        conn.close()


 
if __name__ == '__main__':
    app.run(debug=True)
