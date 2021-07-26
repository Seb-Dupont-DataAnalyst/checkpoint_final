import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import plotly.express as px
from functools import reduce
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Lasso, LassoLarsCV
from sklearn import linear_model
from traitlets.traitlets import default
from xgboost import XGBRegressor
from sklearn import ensemble
from sklearn import metrics
import plotly.graph_objects as go
import collections
from pandas.api.types import CategoricalDtype
from datetime import datetime, date, time
import regex as re
from plotly.subplots import make_subplots
import string
import nltk
from nltk.corpus import stopwords
import PIL.Image
from PIL import Image
import requests
import streamlit.components.v1 as components


st.set_page_config(page_title="Domaine des Croix",
                   page_icon="🍷", layout="wide",
                   initial_sidebar_state="expanded",
                   )

link = "https://github.com/murpi/wilddata/raw/master/wine.zip"
df = pd.read_csv(r"https://raw.githubusercontent.com/Seb-Dupont-DataAnalyst/checkpoint_final/main/df.csv")

link2 = "https://github.com/murpi/wilddata/raw/master/domaine_des_croix.csv"
df_domaine = pd.read_csv(link2)

date_reg = r'([0-9]{4})'

# date_reg = r'([0-9]{4})'
# df['millésime']  = df['title'].str.extract(date_reg, expand=False)
df = df.dropna(subset=['country'])

# st.markdown("""  <style> .reportview-container { background:
#     url("https://wallpaperaccess.com/full/44954.jpg")}
#     </style> """, unsafe_allow_html=True)

# st.set_page_config(layout='wide')

# Titre principale
#components.html("<body style='color:white;font-family:verdana; font-size:60px; border: 2px solid white; text-align: center; padding: 1px'><b>Cinéma Le Creusois</b></body>")
st.markdown('<style>' + open('style.css').read() +
            '</style>', unsafe_allow_html=True)
st.markdown('<body class="title">🍇 Domaine des Croix 🍷</body>',
            unsafe_allow_html=True)

st.sidebar.title("Bienvenue :wine_glass: :grapes:")


#categorie = st.sidebar.radio("Categories", ("The Mission", "The Data", "Clustering", "Optimize a seller profile", "Last words"))


choice = st.sidebar.radio(
    "", ('Présentation', 'Analyse du marché', "Zoom", "Descriptions", 'Pricing via Machine Learning'))



# Création Sidebar avec les différents choix
liste_pays = df['country'].unique().tolist()
liste_pays.insert(0, 'Tous')

st.title('')
st.title('')


#choix_pays = st.selectbox('Select a continent :', liste_pays)

if choice == 'Présentation':

    st.markdown("""  <style> .reportview-container { background:
     url("https://mocah.org/uploads/posts/349986-4k-wallpaper.jpg");
    background-size: cover}
     </style> """, unsafe_allow_html=True)

    components.html("<p style='color:black;font-family:verdana; font-size:20px; text-align: center'><i>Bienvenue sur cette page, nous allons voir ensemble une analyse détaillée du marché des vins. Dans une seconde partie nous aborderons ensemble une proposition de pricing pour vos produits.</i></p>")
    
    

if choice == 'Analyse du marché':

    # choix_pays = st.selectbox('Select countries', liste_pays,
    #                           format_func=lambda x: 'Select a country' if x == '' else x)

    # if choix_pays != 'Tous':

    #     #df = df[df['country'].isin(choix_pays)]
    #     df = df[df['country'] == (choix_pays)]

    # else:

    #     st.warning('Vision globale')

    st.subheader('')

    #st.markdown("<body class='p3'>Quelques graphiques pour l'analyse descriptive :</body>", unsafe_allow_html=True)
    st.title('')

    fig = go.Figure(data=go.Choropleth(
        locations=df.groupby(['code']).count()[
            ['title']].reset_index()['code'],
        z=df.groupby(['code']).count()[['title']].reset_index()['title'],
        # text=df.groupby(['code']).count()[['title']].reset_index()['title'],
        colorscale=px.colors.sequential.Sunsetdark,
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title='Nombre de vins',
    ))
    fig.update_layout(
        dragmode=False,
        title_text='<b> Où les vins sont ils le plus produits ? </b>', title_x=0.5,
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='natural earth'
        )
    )
    fig.update_geos(bgcolor='rgba(0,0,0,0)')

    st.plotly_chart(fig, use_container_width=True)

    trace1 = go.Box(
        y=df['points'],
        name='Notes',
        marker=dict(
            color='royalblue'
        )
    )
    trace2 = go.Box(
        y=df['price'],
        name='Prix',
        marker=dict(
            color='#FF851B'
        ),
        yaxis='y2'
    )
    data = [trace1, trace2]
    layout = go.Layout(
        yaxis=go.layout.YAxis(
            title='Notes',
            zeroline=False
        ),
        yaxis2=go.layout.YAxis(
            side='right',
            title='Prix',
            overlaying='y',
            range=[0, 100],
            zeroline=False),
        boxmode='group'
    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        title='<b>Répartition des notes et des prix</b>', title_x=0.5, showlegend=False)
    fig.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                       'paper_bgcolor': 'rgba(255,255,255,255)', })

    fig.add_annotation(x=1.17, y=df['price'].median(), yref="y2",text=df['price'].median(),showarrow=False, font=dict(color="white", size=14))
    fig.add_annotation(x=-0.17, y=df['points'].median(), yref="y",text=df['points'].median(),showarrow=False, font=dict(color="white", size=14))
              
    st.plotly_chart(fig, use_container_width=True)

    fig = px.bar(df.groupby(['country']).count()[['title']].reset_index().sort_values(
        by='title', ascending=False)[0:10], x='country', y='title', color='country', text='title')
    fig.update_layout(xaxis={'categoryorder': 'total descending'}, title='<b>Répartition des vins par pays (top 10)</b>',
                      title_x=0.5, title_font_family="Verdana", showlegend=False)
    fig.update_layout(xaxis_title="Pays",
                      yaxis_title="Nb vins", showlegend=False)
    fig.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                       'paper_bgcolor': 'rgba(255,255,255,255)', })
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    st.plotly_chart(fig, use_container_width=True)

    fig3 = px.pie(df.groupby(['country']).count()[['title']].reset_index(
    ), values='title', names='country', labels='title', hole=.5)
    fig3.update_layout(width=800, height=500)
    fig3.update_traces(textposition='inside')
    fig3.update_traces(texttemplate="%{label} <br>%{percent:%f}")
    fig3.update_layout(title='<b>Répartition des vins par pays</b>',
                       title_x=0.5, title_font_family="Verdana", showlegend=False)

    st.plotly_chart(fig3, use_container_width=True)

    df_country_mean = df.groupby(['country']).mean(
    )[['points']].reset_index().sort_values(by='points', ascending=False)
    df_country_count = df.groupby(['country']).count(
    )[['points']].reset_index().sort_values(by='points', ascending=False)
    df_country = df_country_mean.merge(
        df_country_count, how='inner', on='country')
    df_country.columns = ['country', 'mean', 'count']
    df_country.sort_values(by='mean', ascending=False).head(10)

    fig = go.Figure(data=[
        go.Bar(name='Notes', x=df_country.sort_values(by='mean', ascending=False).head(10)['country'], y=df_country.sort_values(by='mean', ascending=False).head(10)['mean'], yaxis='y',
               offsetgroup=1, text=df_country.sort_values(by='mean', ascending=False).head(10)['mean'], marker={'color': ['lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen']}),
        go.Bar(name='Nombre',  x=df_country.sort_values(by='mean', ascending=False).head(10)['country'], y=df_country.sort_values(by='mean', ascending=False).head(10)['count'], yaxis='y2',
               offsetgroup=2, text=df_country.sort_values(by='mean', ascending=False).head(10)['count'], marker={'color': ['yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow']})
    ],
        layout={
        'yaxis': {'title': 'Notes Moyennes'},
        'yaxis2': {'title': 'Nombre de vins', 'overlaying': 'y', 'side': 'right'}
    })
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                       'paper_bgcolor': 'rgba(255,255,255,255)', })
    fig.update_layout(title='<b>Top 10 pays par notes</b>',
                      title_x=0.5, title_font_family="Verdana")
    fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')

    fig1 = go.Figure(data=[
        go.Bar(name='Notes', x=df_country.sort_values(by='count', ascending=False).head(10)['country'], y=df_country.sort_values(by='count', ascending=False).head(10)['mean'], yaxis='y',
               offsetgroup=1, text=df_country.sort_values(by='count', ascending=False).head(10)['mean'], marker={'color': ['lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen', 'lightgreen']}),
        go.Bar(name='Nombre',  x=df_country.sort_values(by='count', ascending=False).head(10)['country'], y=df_country.sort_values(by='count', ascending=False).head(10)['count'], yaxis='y2',
               offsetgroup=2, text=df_country.sort_values(by='count', ascending=False).head(10)['count'], marker={'color': ['yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow']})
    ],
        layout={
        'yaxis': {'title': 'Notes moyennes'},
        'yaxis2': {'title': 'Nombre de vins', 'overlaying': 'y', 'side': 'right'}
    })
    # Change the bar mode
    fig1.update_layout(barmode='group')
    fig1.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                        'paper_bgcolor': 'rgba(255,255,255,255)', })
    fig1.update_layout(title='<b>Top 10 pays par nombre</b>',
                       title_x=0.5, title_font_family="Verdana")
    fig1.update_traces(texttemplate='%{text:.3s}', textposition='outside')

    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig1, use_container_width=True)

    df_variety_mean = df.groupby(['variety']).mean(
    )[['points']].reset_index().sort_values(by='points', ascending=False)
    df_variety_count = df.groupby(['variety']).count(
    )[['points']].reset_index().sort_values(by='points', ascending=False)
    df_variety = df_variety_mean.merge(
        df_variety_count, how='inner', on='variety')
    df_variety.columns = ['variety', 'mean', 'count']

    fig = go.Figure(data=[
        go.Bar(name='Notes', x=df_variety.sort_values(by='mean', ascending=False).head(10)['variety'], y=df_variety.sort_values(by='mean', ascending=False).head(10)['mean'], yaxis='y',
               offsetgroup=1, text=df_variety.sort_values(by='mean', ascending=False).head(10)['mean'],
               marker={'color': ['lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue']}),
        go.Bar(name='Nombre',  x=df_variety.sort_values(by='mean', ascending=False).head(10)['variety'], y=df_variety.sort_values(by='mean', ascending=False).head(10)['count'], yaxis='y2',
               offsetgroup=2, text=df_variety.sort_values(by='mean', ascending=False).head(10)['count'],
               marker={'color': ['yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow']})
    ],
        layout={
        'yaxis': {'title': 'Notes moyennes'},
        'yaxis2': {'title': 'Nombre de vins', 'overlaying': 'y', 'side': 'right'}
    })
    # Change the bar mode
    fig.update_layout(barmode='group')
    fig.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                       'paper_bgcolor': 'rgba(255,255,255,255)', })
    fig.update_layout(title='<b>Top 10 cépages par notes</b>',
                      title_x=0.5, title_font_family="Verdana")
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    fig1 = go.Figure(data=[
        go.Bar(name='Notes', x=df_variety.sort_values(by='count', ascending=False).head(10)['variety'], y=df_variety.sort_values(by='count', ascending=False).head(10)['mean'], yaxis='y',
               offsetgroup=1, text=df_variety.sort_values(by='count', ascending=False).head(10)['mean'],
               marker={'color': ['lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue', 'lightblue']}),
        go.Bar(name='Nombre',  x=df_variety.sort_values(by='count', ascending=False).head(10)['variety'], y=df_variety.sort_values(by='count', ascending=False).head(10)['count'], yaxis='y2',
               offsetgroup=2, text=df_variety.sort_values(by='count', ascending=False).head(10)['count'],
               marker={'color': ['yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow', 'yellow']})
    ],
        layout={
        'yaxis': {'title': 'Notes moyennes'},
        'yaxis2': {'title': 'Nombre de vins', 'overlaying': 'y', 'side': 'right'}
    })
    # Change the bar mode
    fig1.update_layout(barmode='group')
    fig1.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                        'paper_bgcolor': 'rgba(255,255,255,255)', })
    fig1.update_layout(title='<b>Top 10 cépages par nombre</b>',
                       title_x=0.5, title_font_family="Verdana")
    fig1.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    st.plotly_chart(fig, use_container_width=True)
    st.plotly_chart(fig1, use_container_width=True)

    fig1 = go.Figure(data=[
        go.Bar(name='Notes', x=df.groupby(['millésime']).mean()[['points']].reset_index()['millésime'], y=df.groupby(['millésime']).mean()[['points']].reset_index()['points'], yaxis='y',
               offsetgroup=1, text=df.groupby(['millésime']).mean()[['points']].reset_index()['points'],
               ),
        go.Bar(name='Nombre', x=df.groupby(['millésime']).count()[['points']].reset_index()['millésime'], y=df.groupby(['millésime']).count()[['points']].reset_index()['points'], yaxis='y2',
               offsetgroup=2, text=df.groupby(['millésime']).count()[['points']].reset_index()['points'],
               ),
    ],
        layout={
        'yaxis': {'title': 'Notes moyennes'},
        'yaxis2': {'title': 'Nombre de vins', 'overlaying': 'y', 'side': 'right'}
    })
    # Change the bar mode
    fig1.update_layout(barmode='group')
    fig1.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                        'paper_bgcolor': 'rgba(255,255,255,255)', })
    fig1.update_layout(title='<b>Evolution des notes et du nombre de vins par millésime</b>',
                       title_x=0.5, title_font_family="Verdana")
    fig1.update_traces(texttemplate='%{text:.2s}', textposition='outside')
    fig1.update_xaxes(title='Millésimes', range=[1990, 2022])
    fig1.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                       'paper_bgcolor': 'rgba(255,255,255,255)', })

    st.plotly_chart(fig1, use_container_width=True)

if choice == 'Zoom':



    sub_choice= st.sidebar.radio("Zoom", ('Cépages', 'Province'))    

    if sub_choice == 'Cépages' :
        liste_cepage = df['variety'].unique().tolist()
        # liste_cepage.insert(0, 'Tous')

        st.title('')
        st.title('')


        choix_cepage = st.selectbox('Sélectionner un cépage :', liste_cepage)
        df_cepage = df[df['variety'] == choix_cepage]
        df_hors_Pinot = df[df['variety'] != choix_cepage]

        fig = go.Figure(data=go.Choropleth(
            locations=df_cepage.groupby(['code']).count()[
                ['title']].reset_index()['code'],
            z=df_cepage.groupby(['code']).count()[['title']].reset_index()['title'],
            # text=df.groupby(['code']).count()[['title']].reset_index()['title'],
            colorscale=px.colors.sequential.Sunsetdark,
            autocolorscale=False,
            reversescale=False,
            marker_line_color='darkgray',
            marker_line_width=0.5,
            colorbar_title='Nombre de vins',
        ))
        fig.update_layout(
            dragmode=False,
            title_text='<b> Où le ' + choix_cepage + ' est il le plus produit ? </b>', title_x=0.5,
            geo=dict(
                showframe=False,
                showcoastlines=False,
                projection_type='natural earth'
            )
        )
        fig.update_geos(bgcolor='rgba(0,0,0,0)')

        st.plotly_chart(fig, use_container_width=True)

        trace1 = go.Box(
        y=df_cepage['points'],
        name=choix_cepage +' ',
        marker=dict(
            color='royalblue'
        )
        )

        trace2 = go.Box(
        y=df_hors_Pinot['points'],
        name='Autres',
        marker=dict(
            color='cyan'
        )
        )

        trace3 = go.Box(
        y=df_cepage['price'],
        name= choix_cepage,
        marker=dict(
            color='royalblue'
        ),
        yaxis='y2'


        )
        trace4 = go.Box(
        y=df_hors_Pinot['price'],
        name='Autres ',
        marker=dict(
            color='cyan'
        ),
        yaxis='y2'
        )
        data = [trace1, trace2, trace3, trace4]
        layout = go.Layout(
        yaxis=go.layout.YAxis(
            title='notes',
            zeroline=False
        ),
        yaxis2=go.layout.YAxis(
            side='right',
            title='prix',
            overlaying='y',
            range=[0, 100],
            zeroline=False),
        boxmode='group'
        )

        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(showlegend = False)
        fig.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                            'paper_bgcolor': 'rgba(255,255,255,255)', })
        #fig.update_xaxes(color='white')
        fig.add_annotation(
            x=0.3,
            y=100,
            xref="x",
            yref="y",
            text="Notes",
            showarrow=False,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#ffffff"
            ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            ax=20,
            ay=-30,
            borderwidth=2,
            borderpad=2,
            bgcolor='#257d98',
            opacity=0.8
            )

        fig.add_annotation(
            x=2.7,
            y=100,
            xref="x",
            yref="y",
            text="Prix",
            showarrow=False,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#ffffff"
            ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            ax=20,
            ay=-30,
            borderwidth=2,
            borderpad=2,
            bgcolor='#4ac8a4',
            opacity=0.8
            )
        
        fig.add_annotation(x=2.1, y=df_cepage['price'].median(), yref="y2",text=df_cepage['price'].median(),showarrow=False, font=dict(color="black", size=14))
        fig.add_annotation(x=-0.25, y=df_cepage['points'].median(), yref="y",text=df_cepage['points'].median(),showarrow=False, font=dict(color="black", size=14))
        fig.add_annotation(x=3.26, y=df_hors_Pinot['price'].median(), yref="y2",text=df_hors_Pinot['price'].median(),showarrow=False, font=dict(color="black", size=14))
        fig.add_annotation(x=0.9, y=df_hors_Pinot['points'].median(), yref="y",text=df_hors_Pinot['points'].median(),showarrow=False, font=dict(color="black", size=14))
        fig.update_layout(title='<b>Comparaison : ' + choix_cepage + ' vs autres</b>',
                            title_x=0.5, title_font_family="Verdana")
        st.plotly_chart(fig, use_container_width=True)


        fig1 = go.Figure(data=[
        go.Bar(name='Notes moyennes', x=df_cepage.groupby(['millésime']).mean()[['points']].reset_index()['millésime'], y=df_cepage.groupby(['millésime']).mean()[['points']].reset_index()['points'], yaxis='y', 
            offsetgroup=1, text=df_cepage.groupby(['millésime']).mean()[['points']].reset_index()['points'],
            ),
        go.Bar(name='Nombre de vins', x=df_cepage.groupby(['millésime']).count()[['points']].reset_index()['millésime'], y=df_cepage.groupby(['millésime']).count()[['points']].reset_index()['points'], yaxis='y2', 
            offsetgroup=2, text=df_cepage.groupby(['millésime']).count()[['points']].reset_index()['points'],
            ),
        ],
        layout={
            'yaxis': {'title': 'Notes moyennes'},
            'yaxis2': {'title': 'Nombre de vins', 'overlaying': 'y', 'side': 'right'}
        })
        # Change the bar mode
        fig1.update_layout(barmode='group')
        fig1.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                            'paper_bgcolor': 'rgba(255,255,255,255)', })
        fig1.update_layout(title='<b>Evolution des notes et du nombre de vins par millésime</b>',
                            title_x=0.5, title_font_family="Verdana")
        fig1.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig1.update_xaxes(title='Millésimes ' + choix_cepage, range=[2000,2022])
        fig1.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                            'paper_bgcolor': 'rgba(255,255,255,255)', })

        st.plotly_chart(fig1, use_container_width=True)


    if sub_choice == 'Province' :
        liste_province = df['province'].unique().tolist()
        # liste_cepage.insert(0, 'Tous')

        st.title('')
        st.title('')


        choix_province = st.selectbox('Sélectionner une province :', liste_province)
        df_province = df[df['province'] == choix_province]
        df_hors_province = df[df['province'] != choix_province]

        
        trace1 = go.Box(
        y=df_province['points'],
        name=choix_province +' ',
        marker=dict(
            color='royalblue'
        )
        )

        trace2 = go.Box(
        y=df_hors_province['points'],
        name='Autres',
        marker=dict(
            color='cyan'
        )
        )

        trace3 = go.Box(
        y=df_province['price'],
        name= choix_province,
        marker=dict(
            color='royalblue'
        ),
        yaxis='y2'


        )
        trace4 = go.Box(
        y=df_hors_province['price'],
        name='Autres ',
        marker=dict(
            color='cyan'
        ),
        yaxis='y2'
        )
        data = [trace1, trace2, trace3, trace4]
        layout = go.Layout(
        yaxis=go.layout.YAxis(
            title='notes',
            zeroline=False
        ),
        yaxis2=go.layout.YAxis(
            side='right',
            title='prix',
            overlaying='y',
            range=[0, 100],
            zeroline=False),
        boxmode='group'
        )

        fig = go.Figure(data=data, layout=layout)
        fig.update_layout(showlegend = False)
        fig.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                            'paper_bgcolor': 'rgba(255,255,255,255)', })
        #fig.update_xaxes(color='white')
        fig.add_annotation(
            x=0.3,
            y=100,
            xref="x",
            yref="y",
            text="Notes",
            showarrow=False,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#ffffff"
            ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            ax=20,
            ay=-30,
            borderwidth=2,
            borderpad=2,
            bgcolor='#257d98',
            opacity=0.8
            )

        fig.add_annotation(
            x=2.7,
            y=100,
            xref="x",
            yref="y",
            text="Prix",
            showarrow=False,
            font=dict(
                family="Courier New, monospace",
                size=16,
                color="#ffffff"
            ),
            align="center",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            ax=20,
            ay=-30,
            borderwidth=2,
            borderpad=2,
            bgcolor='#4ac8a4',
            opacity=0.8
            )
        fig.update_layout(title='<b>Comparaison : ' + choix_province + ' vs autres</b>',
                            title_x=0.5, title_font_family="Verdana")
        st.plotly_chart(fig, use_container_width=True)


        fig1 = go.Figure(data=[
        go.Bar(name='Notes moyennes', x=df_province.groupby(['millésime']).mean()[['points']].reset_index()['millésime'], y=df_province.groupby(['millésime']).mean()[['points']].reset_index()['points'], yaxis='y', 
            offsetgroup=1, text=df_province.groupby(['millésime']).mean()[['points']].reset_index()['points'],
            ),
        go.Bar(name='Nombre de vins', x=df_province.groupby(['millésime']).count()[['points']].reset_index()['millésime'], y=df_province.groupby(['millésime']).count()[['points']].reset_index()['points'], yaxis='y2', 
            offsetgroup=2, text=df_province.groupby(['millésime']).count()[['points']].reset_index()['points'],
            ),
        ],
        layout={
            'yaxis': {'title': 'Notes moyennes'},
            'yaxis2': {'title': 'Nombre de vins', 'overlaying': 'y', 'side': 'right'}
        })
        # Change the bar mode
        fig1.update_layout(barmode='group')
        fig1.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                            'paper_bgcolor': 'rgba(255,255,255,255)', })
        fig1.update_layout(title='<b>Evolution des notes et du nombre de vins par millésime</b>',
                            title_x=0.5, title_font_family="Verdana")
        fig1.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        fig1.update_xaxes(title='Millésimes ' + choix_province, range=[2000,2022])
        fig1.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                            'paper_bgcolor': 'rgba(255,255,255,255)', })

        st.plotly_chart(fig1, use_container_width=True)

if choice == 'Descriptions' :

    st.markdown("""  <style> .reportview-container { background:
     url("https://ak.picdn.net/shutterstock/videos/2715950/thumb/4.jpg");
    background-size: cover}
     </style> """, unsafe_allow_html=True)
    

    #st.markdown("<body class='p3'>Quelques graphiques pour l'analyse des mots utilisés dans les descriptions :</body>", unsafe_allow_html=True)
    st.header('')

    Top = [('wine', 80297),
    ('flavors', 62748),
    ('fruit', 49921),
    ('aromas', 39600),
    ('palate', 38495),
    ('acidity', 34980),
    ('finish', 34933),
    ('tannins', 30857),
    ('drink', 30311),
    ('cherry', 29289)]

    Top_pinot = [('wine', 8569),
    ('flavors', 6253),
    ('cherry', 6123),
    ('fruit', 5594),
    ('pinot', 3975),
    ('acidity', 3575),
    ('red', 3519),
    ('tannins', 3195),
    ('finish', 2981),
    ('black', 2962)]

    Top_burgundy = [('wine', 5171),
    ('acidity', 2404),
    ('drink', 2136),
    ('ripe', 1787),
    ('fruit', 1665),
    ('fruits', 1618),
    ('flavors', 1396),
    ('rich', 1097),
    ('tannins', 1068),
    ('texture', 956)]

    df_Top  = pd.DataFrame(Top)
    df_Top_pinot  = pd.DataFrame(Top_pinot)
    df_Top_burgundy  = pd.DataFrame(Top_burgundy)

    fig = make_subplots(rows=1, cols=3, specs=[[{'type':'domain'}, {'type':'domain'}, {'type':'domain'}]])
    fig.add_trace(go.Pie(labels=df_Top_burgundy[0], values=df_Top_burgundy[1], name="Burgundy", hole=0.5, marker_colors=px.colors.qualitative.Plotly),
                1, 3)
        
    fig.add_trace(go.Pie(labels=df_Top_pinot[0], values=df_Top_pinot[1], name="Pinot", hole=0.5, marker_colors=px.colors.qualitative.Plotly),
                1, 2)
    fig.add_trace(go.Pie(labels=df_Top[0], values=df_Top[1], name="Global", hole=0.5, marker_colors=px.colors.qualitative.Plotly),
                1, 1)
    fig.update_layout(title='<b>Mots les plus utilisés dans les descriptions</b>',
                        title_x=0.5, title_font_family="Verdana", title_font_color = 'black',showlegend=False)
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)','paper_bgcolor': 'rgba(0,0,0,0)'})
    fig.update_traces(texttemplate = "%{label} <br>%{percent:%f}")
    fig.update_layout(
    annotations=[dict(text='Global', x=0.125, y=0.5, font_size=20, showarrow=False, font=dict(color="black", size=12)),
             dict(text='Pinot Noir', x=0.5, y=0.5, font_size=20, showarrow=False, font=dict(color="black", size=12)),
              dict(text='Burgundy', x=0.89, y=0.5, font_size=20, showarrow=False, font=dict(color="black", size=12))])

    st.plotly_chart(fig, use_container_width=True)

    #st.image("https://github.com/Seb-Dupont-DataAnalyst/checkpoint_final/blob/main/graph%20NLP.JPG?raw=true")

    st.subheader('')

    st.image("https://github.com/Seb-Dupont-DataAnalyst/checkpoint_final/blob/main/graph_NLP_2.JPG?raw=true", width = 1300)

    st.header('')

    df['desc_len'] = df['description'].apply(lambda x : len(x))

    fig = px.bar(df.groupby(['millésime']).mean()[['desc_len']].reset_index(), x='millésime', y='desc_len', color='millésime', text = 'desc_len')
    fig.update_xaxes(title='Millésimes', range=[1990, 2022])
    fig.update_layout(title='<b>Evolution de la longueur des descriptions</b>',
                      title_x=0.5, title_font_family="Verdana", title_font_color = 'black', showlegend=False)
    fig.layout.coloraxis.showscale = False
    fig.update_layout(xaxis_title="Millésimes",
                      yaxis_title="Nb mots", showlegend=False)
    fig.update_layout({'plot_bgcolor': 'rgba(0,0,0,0)','paper_bgcolor': 'rgba(0,0,0,0)'})
    # fig.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
    #                    'paper_bgcolor': 'rgba(255,255,255,255)', })
    fig.update_yaxes(showgrid=False, gridcolor='black')
    fig.update_xaxes(showgrid=False, gridcolor='black')
    fig.update_yaxes(color = 'black')
    fig.update_xaxes(color = 'black')
    fig.update_traces(texttemplate='%{text:.3s}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

if choice == 'Pricing via Machine Learning' :

    # st.markdown("""  <style> .reportview-container { background:
    #  url("https://ak.picdn.net/shutterstock/videos/2715950/thumb/4.jpg");
    # background-size: cover}
    #  </style> """, unsafe_allow_html=True)

    #epuration
    
    df_top10_notes = df[df['country'].isin(['United Kingdom', 'India', 'Austria', 'Germany', 'Canada', 'Hungary', 'China', 'France', 'Luxembourg', 'Australia']) ]
    df_top10_country_cepage = df[(df['country'].isin(['US', 'France', 'Italy', 'Spain', 'Portugal', 'Chile', 'Argentina', 'Austria', 'Germany', 'New Zealand'])) & 
                             (df['variety'].isin(['Pinot Noir', 'Chardonnay', 'Cabernet Sauvignon', 'Red Blend', 'Bordeaux-style Red Blend', 'Riesling', 'Sauvignon Blanc', 'Syrah', 'Rosé', 'New Merlot']))]
    
    df_country_mean = df.groupby(['country']).mean()[['points']].reset_index().sort_values(by='points', ascending=False)
    df_country_count = df.groupby(['country']).count()[['points']].reset_index().sort_values(by='points', ascending=False)
    df_country = df_country_mean.merge(df_country_count, how='inner', on='country')
    df_country.columns = ['country', 'mean', 'count']
    df_country.sort_values(by='mean', ascending=False).head(10)

    df_variety_mean = df.groupby(['variety']).mean()[['points']].reset_index().sort_values(by='points', ascending=False)
    df_variety_count = df.groupby(['variety']).count()[['points']].reset_index().sort_values(by='points', ascending=False)
    df_variety = df_variety_mean.merge(df_variety_count, how='inner', on='variety')
    df_variety.columns = ['variety', 'mean', 'count']

    df_price_clean = df.dropna(subset=['price', 'millésime'])
    df_price_clean['price'] = df_price_clean['price'].astype(int)

    df_top10_country_cepage_clean = df_top10_country_cepage.dropna(subset=['millésime', 'price'])
    df_top10_country_cepage_clean['price'] = df_top10_country_cepage_clean['price'].astype(int)
    df_top10_country_cepage_clean['millésime'] = df_top10_country_cepage_clean['millésime'].astype(int)



    df_top10_country_cepage_clean['province_fact'] = df_top10_country_cepage_clean['province'].factorize()[0]
    df_top10_country_cepage_clean['variety_fact'] = df_top10_country_cepage_clean['variety'].factorize()[0]
    df_top10_country_cepage_clean['country_fact'] = df_top10_country_cepage_clean['country'].factorize()[0]


    liste_cepage = df_top10_country_cepage_clean['variety'].unique().tolist()
    liste_cepage.insert(0, 'Tous')

    st.title('')
 


    choix_cepage = st.selectbox('Sélectionner un cépage :', liste_cepage)

    liste_province = df_top10_country_cepage_clean['province'].unique().tolist()
    liste_province.insert(0, 'Tous')

  


    choix_province = st.selectbox('Sélectionner une province :', liste_province)

    saisie_taux = st.number_input("Saisissez un taux pour l'hypothèse intermédiaire :", 0.75)

    st.subheader('')




    y = df_price_clean['price']

    X = df_price_clean[['points','millésime']]


    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42, train_size = 0.75)

    model_gbr = ensemble.GradientBoostingRegressor(min_samples_split= 6, min_samples_leaf= 49, loss= 'ls').fit(X_train, y_train)
    accuracy_gbr = model_gbr.score(X_train, y_train)
    accuracy_gbr_test = model_gbr.score(X_test, y_test)


    df_domaine['province_fact'] = 3
    df_domaine['variety_fact'] = df_domaine['variety'].apply(lambda x : 3 if x=='Pinot Noir' else 4)
    df_domaine['country_fact'] = 3
    df_domaine['millésime']  = df_domaine['title'].str.extract(date_reg, expand=False)
    df_domaine['millésime'] = df_domaine['millésime'].astype(int)
    X = df_domaine[['points',	'millésime']]
    

    df_domaine['price'] = model_gbr.predict(X).round()




    y2 = df_top10_country_cepage_clean['price']

    X2 = df_top10_country_cepage_clean[['country_fact', 'points', 'millésime', 'province_fact', 'variety_fact']]


    X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, random_state = 42, train_size = 0.75)


    model_gbr2 = ensemble.GradientBoostingRegressor(min_samples_split= 6, min_samples_leaf= 49, loss= 'ls').fit(X2_train, y2_train)
    accuracy_gbr2 = model_gbr2.score(X2_test, y2_test)
 

    X2 = df_domaine[['country_fact', 'points', 'millésime', 'province_fact', 'variety_fact']]
    df_domaine['price 2'] = model_gbr2.predict(X2).round()
    if saisie_taux:
        df_domaine['price 3'] = df_domaine['price 2'] * saisie_taux
    if choix_cepage == 'Tous':
        st.warning('Veuillez choisir un cépage')
        

    elif choix_cepage != 'Tous':
        if choix_province == 'Tous':
            box_cepage = df[(df['variety'] == choix_cepage) & (df['points'] > 90)]
            box_cepage_france = df[(df['variety'] == choix_cepage) & (df['points'] > 90) & (df['code'] =='FRA')]

            trace1 = go.Box(
            y=df_domaine['price'],
            name='Domaine des Croix 1',
            marker=dict(
                color='burlywood'
            )
            )
            trace2 = go.Box(
                y=df_domaine['price 3'],
                name='Domaine des Croix 2',
                marker=dict(
                    color='brown'
                ))

            trace3 = go.Box(
                y=df_domaine['price 2'],
                name='Domaine des Croix 3',
                marker=dict(
                    color='tomato'
                )
            )
            trace4 = go.Box(
                y=box_cepage_france['price'],
                name=choix_cepage + ' France',
                marker=dict(
                    color='royalblue'
                ),

            )

            trace5 = go.Box(
                y=box_cepage['price'],
                name=choix_cepage +  ' ' +choix_province,
                marker=dict(
                    color='lightgreen'
                ),

            )
            data = [trace1, trace2, trace3, trace4, trace5]
            layout = go.Layout(
            yaxis=go.layout.YAxis(
                    side='left',
                    title='prix',
                    overlaying='y',
                    range=[0, 200],
                    zeroline=False),
                boxmode='group'
            )

            fig = go.Figure(data=data, layout=layout)
            fig.update_layout(showlegend = False)
            fig.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                                'paper_bgcolor': 'rgba(255,255,255,255)', })
            st.plotly_chart(fig, use_container_width=True)

        
            
        if choix_province != 'Tous': 
                box_cepage = df[(df['variety'] == choix_cepage) & (df['points'] > 90) & (df['province'] == choix_province)]
                box_cepage_france = df[(df['variety'] == choix_cepage) & (df['points'] > 90) & (df['code'] =='FRA')]
                
                trace1 = go.Box(
                y=df_domaine['price'],
                name='Domaine des Croix 1',
                marker=dict(
                    color='burlywood'
                )
                )

                trace2 = go.Box(
                y=df_domaine['price 3'],
                name='Domaine des Croix 2',
                marker=dict(
                    color='brown'
                )
                )

                trace3 = go.Box(
                    y=df_domaine['price 2'],
                    name='Domaine des Croix 3',
                    marker=dict(
                        color='tomato'
                    )
                )

                trace4 = go.Box(
                y=box_cepage_france['price'],
                name=choix_cepage + ' France',
                marker=dict(
                    color='royalblue'
                ),

            )
                trace5 = go.Box(
                    y=box_cepage['price'],
                    name=choix_cepage + ' ' + choix_province,
                    marker=dict(
                        color='lightgreen'
                    ),

                )
                data = [trace1, trace2, trace3, trace4, trace5]
                layout = go.Layout(
                yaxis=go.layout.YAxis(
                        side='left',
                        title='prix',
                        overlaying='y',
                        range=[0, 320],
                        zeroline=False),
                    boxmode='group'
                )

                fig = go.Figure(data=data, layout=layout)
                fig.update_layout(title = "<b>Comparaison du pricing avec l'existant</b>", title_x = 0.5, showlegend = False)
                fig.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                                    'paper_bgcolor': 'rgba(255,255,255,255)', })
                st.plotly_chart(fig, use_container_width=True)

        choix_hypothese = st.selectbox('Sélectionner une hypothèse de pricing',['Hypothèse basse', 'Hypothèse intermédiaire', 'Hypothèse haute'])

        if choix_hypothese == 'Hypothèse basse':
            fig = px.bar(df_domaine, x='price', y='title',orientation='h', color = 'title', text='price')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, title='<b>Pricing des différents vins</b>',
                        title_x=0.5, title_font_family="Verdana", showlegend=False)  
            fig.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                                        'paper_bgcolor': 'rgba(255,255,255,255)', })
            fig.update_traces(texttemplate='&#36;%{text:.3s}', textposition='outside')
            fig.update_xaxes(title='Prix')
            fig.update_yaxes(title='Vins')
            fig.update_layout(xaxis_tickprefix = '$')
            st.plotly_chart(fig, use_container_width=True)    

        if choix_hypothese == 'Hypothèse haute':
            fig = px.bar(df_domaine, x='price 2', y='title',orientation='h', color = 'title', text='price 2')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, title='<b>Pricing des différents vins</b>',
                        title_x=0.5, title_font_family="Verdana", showlegend=False)  
            fig.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                                        'paper_bgcolor': 'rgba(255,255,255,255)', })
            fig.update_traces(texttemplate='&#36;%{text:.3s}', textposition='outside')
            fig.update_xaxes(title='Prix')
            fig.update_yaxes(title='Vins')
            fig.update_layout(xaxis_tickprefix = '$')
            st.plotly_chart(fig, use_container_width=True)   

        if choix_hypothese == 'Hypothèse intermédiaire':
            fig = px.bar(df_domaine, x='price 3', y='title',orientation='h', color = 'title', text='price 3')
            fig.update_layout(yaxis={'categoryorder': 'total ascending'}, title='<b>Pricing des différents vins</b>',
                        title_x=0.5, title_font_family="Verdana", showlegend=False)  
            fig.update_layout({'plot_bgcolor': 'rgba(255,255,255,255)',
                                        'paper_bgcolor': 'rgba(255,255,255,255)', })
            fig.update_traces(texttemplate='&#36;%{text:.3s}', textposition='outside')
            fig.update_xaxes(title='Prix')
            fig.update_yaxes(title='Vins')
            fig.update_layout(xaxis_tickprefix = '$')
            st.plotly_chart(fig, use_container_width=True) 
            

        
