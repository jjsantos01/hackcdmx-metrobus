import pandas as pd
import xgboost as xgb
import geopy.distance
from flask import Flask, request, url_for, render_template
import plotly
import plotly.graph_objs as go
import json
import os


def get_t_estimado(est_o, est_d, h, m):
    time = pd.datetime(2018, 11, 14, h, m)
    sentido = get_sentido(est_o, est_d)
    pares = [[i, i+sentido*1] for i in range(est_o, est_d, sentido)]
    resultado = dict()
    for j in pares:
        datos_pred = pd.DataFrame([
                {'distancia': get_distancia(j[0], j[1]),
                 'est': j[0],
                 't_segm': get_t_segm(time.hour, time.minute),
                 'sentido': sentido,
                 'intersecciones': get_intersecciones(j[0], j[1]),
                 'reciente': get_reciente(j[0], j[1], time.hour, time.minute),
                 'n_estaciones': get_n_estaciones(j[0], j[1]),
                 'precip': get_precip(time.hour),
                 'trafico': get_trafico(j[0], j[1], time.hour, time.minute)}]
                )[x_vars]
        x_test = xgb.DMatrix(datos_pred)
        t_pred = modelo.predict(x_test)[0]
        resultado[j[0]] = t_pred
        time += pd.Timedelta(t_pred, unit='s')
    out = pd.Series(resultado, name='tiempo').clip(lower=30, upper=600)
    return out


def get_trafico(est_o, est_d, h, m):
    t_segm = get_t_segm(h, m)
    li, lu = sorted([est_o, est_d])
    trafico = t_maps.query(f'est_origen>={li} & est_destino<={lu} & t_segm=={t_segm}').eval('duracion_en_trafico-duracion', inplace=False).sum()
    return trafico


def get_precip(h):
    cl = clima.set_index('hora')
    out = cl.loc[h, 'precip']
    return out


def get_reciente(est_o, est_d, h, m):
    """Calcula velocidad promedio de los viajes en las estaciones subsiguientes que van en el mismo sentido
    durante la última media hora"""
    interv1 = max(pd.datetime(2018, 11, 14, 16, 46), pd.datetime(2018, 11, 14, h, m))
    interv0 = interv1 - pd.Timedelta(120, unit='m')
    myest = est_o
    mysigno = '>=' if get_sentido(est_o, est_d)==1 else '<='
    mysent = get_sentido(est_o, est_d)
    out = mb.query(f'"{interv0}"<=fecha<="{interv1}" & est{mysigno}{myest} & sentido=={mysent}')['velocidad']\
        .replace([pd.np.inf, -pd.np.inf], pd.np.nan)\
        .mean()
    return out


def get_n_estaciones(est_o, est_d):
    out = abs(est_o-est_d)
    return out


def get_sentido(est_o, est_d):
    out = 1 if est_d>est_o else -1
    return out


def get_t_segm(h, m):
    out = min(max(97, 6 * h + m//10), 143)
    return out


def get_distancia(est_o, est_d):
    """Calcula la distancia euclidiana para llegar de est1 a est 2"""
    df = estaciones_mb.set_index('est')
    dist = geopy.distance.distance(reversed(df.loc[est_o, 'coord']), reversed(df.loc[est_d, 'coord'])).km
    return dist


def get_intersecciones(est_o, est_d):
    est_o, est_d = sorted([est_o, est_d])
    out = intersecciones.query(f'est_o>={est_o} & est_d<={est_d}')['intersecciones'].sum()
    return out


def get_graph():
    t_estimado = get_t_estimado(valores['est_o'], valores['est_d'], valores['h'], valores['m'])
    t_total = round(t_estimado.sum() / 60, 2)
    est_o = dicc_nom_est[valores["est_o"]]
    est_d = dicc_nom_est[valores["est_d"]]
    sentido = 1 if valores["est_d"] > valores["est_o"] else -1
    valores['t_total'] = f'El tiempo esperado de su viaje de {est_o} a {est_d} es de {t_total} minutos.\n'\
                        f'A continuación se muestra el tiempo estimado (en segundos) entre estaciones:'
    # init a basic bar chart:
    data = [go.Bar(
        x=[dicc_nom_est[i+1*sentido] for i in t_estimado.index.tolist()],
        y=t_estimado.tolist()
    )]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON


# carga archivos
t_maps = pd.read_pickle('datos/t_gmaps_mb_l1.pkl')
clima = pd.read_pickle('datos/clima.pkl')
mb = pd.read_pickle('datos/df_mb_l1.pkl')
estaciones_mb = pd.read_pickle('datos/estaciones_mb.pkl')
intersecciones = pd.read_pickle('datos/intersecciones.pkl')
modelo = xgb.Booster({'nthread': 4})
modelo.load_model('datos/0001.model')
x_vars = ['distancia', 'est', 't_segm', 'sentido', 'intersecciones', 'reciente', 'n_estaciones', 'precip', 'trafico']
# variables gráfica
dicc_est = estaciones_mb.query('est!=15').set_index('Nombre')['est'].to_dict()
dicc_nom_est = estaciones_mb.set_index('est')['Nombre'].to_dict()
opc_est = list(dicc_est)
horas = list(map(str, range(17, 24)))
minutos = list(map(str, range(69)))

# crea Flask object
application = Flask(__name__)
valores = dict(est_o=0, est_d=10, h=17, m=0)


@application.route('/', methods=['GET'])
def index():
    valores['est_o'] = 0
    valores['est_d'] = 46
    valores['h'] = 17
    valores['m'] = 30
    graphJSON = get_graph()
    selectores = render_template('selectores.html', opc_est=opc_est,
                                 est_o=dicc_nom_est[valores['est_o']],
                                 est_d=dicc_nom_est[valores['est_d']],
                                 horas=horas, h=str(valores['h']), minutos=minutos, m=str(valores['m']))
    html = render_template('index.html', selectores=selectores, graphJSON=graphJSON, texto_tiempo=valores['t_total'])
    return html


@application.route('/calcula', methods=['GET'])
def calcula():
    valores['est_o'] = dicc_est[request.args.get('origen')]
    valores['est_d'] = dicc_est[request.args.get('destino')]
    valores['h'] = int(request.args.get('hora'))
    valores['m'] = int(request.args.get('minuto'))
    graphJSON = get_graph()
    selectores = render_template('selectores.html', opc_est=opc_est,
                                 est_o=dicc_nom_est[valores['est_o']],
                                 est_d=dicc_nom_est[valores['est_d']],
                                 horas=horas, h=str(valores['h']), minutos=minutos, m=str(valores['m']))
    html = render_template('index.html', selectores=selectores, graphJSON=graphJSON, texto_tiempo=valores['t_total'])
    return html


if 'Windows' in os.getenv('OS', default=''):
    if __name__ == '__main__':
        application.run(debug=True)

