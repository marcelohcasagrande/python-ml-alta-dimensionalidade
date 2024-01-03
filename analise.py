# Pacotes a serem utilizados.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import RFE, RFECV
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


    #                          # 
    # Leitura da base de dados #
    #                          #

# Lendo base de dados.
resultados_exames = pd.read_csv( 'datasets/exames.csv' )

# Head.
resultados_exames.head()

# Info.
resultados_exames.info()


    #            # 
    # Tratativas #
    #            #

# Verificando missings.
resultados_exames.isna().sum() # Mais de 70% de missing em uma coluna.

# Dropando coluna.
resultados_exames.drop( [ 'exame_33' ], axis = 1, inplace = True )


    #           # 
    # Modelando #
    #           # 

# Fixando semente.
SEED = 123143
np.random.seed( SEED )

# Separando features e resposta.
valores_exames = resultados_exames.drop( [ 'id', 'diagnostico' ], axis = 1 )
diagnostico = resultados_exames[ 'diagnostico' ]

# Separando treino e teste.
treino_x, teste_x, treino_y, teste_y = train_test_split( valores_exames, 
                                                         diagnostico,
                                                         test_size = 0.3 )

# Usando o Random Forest Classifier.
classificador = RandomForestClassifier( n_estimators = 100 )

# Treinando.
classificador.fit( treino_x, treino_y )

# Olhando o resultado para o teste.
print( 'Resultado da classificação: %.2f%%' %( classificador.score( teste_x, teste_y ) * 100 ) )


    #              # 
    # Graficamente #
    #              # 

# Padronizando. A escala dos valores são bem diferentes.
padronizador = StandardScaler()
padronizador.fit( valores_exames )
valores_exames_v1 = padronizador.transform( valores_exames )
valores_exames_v1 = pd.DataFrame( data = valores_exames_v1, columns = valores_exames.keys() ) # colocando em DF.

# Função para gerar gráfico.
def grafico_violino( valores, inicio, fim ):
    dados_plot = pd.concat( [ diagnostico, valores.iloc[ :, inicio:fim ] ], axis = 1 ) # criando dados de plot.
    dados_plot = pd.melt( dados_plot, id_vars = 'diagnostico', var_name = 'exames', value_name = 'valores' ) # usando o melt para deixar o dataframe do jeito que eu quero.
    plt.figure( figsize = ( 10, 10 ) ) # visualizando.
    sns.violinplot( x = 'exames', y = 'valores', hue = 'diagnostico', data = dados_plot, split = True )
    plt.xticks( rotation = 90 ) # rotacionando ticks de X.
    plt.show()


# Vendo valores de 1 até 10.
grafico_violino( valores_exames_v1, 0, 10 )

# Vendo valores de 11 até 20.
grafico_violino( valores_exames_v1, 10, 20 )

# Vendo valores de 21 até 32.
grafico_violino( valores_exames_v1, 20, 32 )

# Dropando colunas constantes.
valores_exames_v2 = valores_exames_v1.drop( [ 'exame_4', 'exame_29' ], axis = 1 )


    #                                                     #
    # Criando função para olhar novamente o classificador #
    #                                                     #
    
# Random Forest para os valores.
def classificar( valores ):
    SEED = 1234
    np.random.seed( SEED )
    treino_x, teste_x, treino_y, teste_y = train_test_split( valores, diagnostico, test_size = 0.3 )
    classificador = RandomForestClassifier( n_estimators = 100 )
    classificador.fit( treino_x, treino_y )
    print( 'Resultado da classificação: %.2f%%' %( classificador.score( teste_x, teste_y ) * 100 ) )
    

# Olhando resultado para a base padronizada e sem 2 variáveis constantes.
classificar( valores_exames_v2 )    


    #                     # 
    # Olhando correlações #
    #                     # 
    
# Mapa de calor do seaborn.
plt.figure( figsize = ( 17, 15 ) )
sns.heatmap( valores_exames_v2.corr(), annot = True, fmt = '.1f' ) # mostrando valores e com 1 casa decimal. 
plt.show()  
    
# Pegando matriz de correlações.
matriz = valores_exames_v2.corr()    
    
# Olhando variáveis altamente correlacionadas.
matriz_v1 = matriz[ matriz > 0.99 ]    

# Calculando a soma e subtraindo 1 da própria correlação.
matriz_v1_soma = matriz_v1.sum() - 1

# Olhando casos altamente correlacionados.
matriz_v1_soma[ matriz_v1_soma > 0 ]

# Eliminando algumas delas.
valores_exames_v3 = valores_exames_v2.drop( [ 'exame_3', 'exame_24' ], axis = 1 )

# Olhando resultado novamente sem as variáveis dropadas.
classificar( valores_exames_v3 )    


    #                                                #
    # Selecionando K melhores features (SelectKBest) #
    #                                                #

# Selecionando usando o SelectKBest.
selecionar_k_melhores = SelectKBest( chi2, k = 5 )

# Retirando colunas já dropadas e deixando sem padronizar, pois chi2 não aceita valores negativos.
valores_exames_v4 = valores_exames.drop( [ 'exame_4', 'exame_29', 'exame_3', 'exame_24' ], axis = 1 )

# Fixando semente.
SEED = 1234
np.random.seed( SEED )

# Separando treino e teste.
treino_x, teste_x, treino_y, teste_y = train_test_split( valores_exames_v4, diagnostico, test_size = 0.3 )

# Treinando usando o Select.
selecionar_k_melhores.fit( treino_x, treino_y )

# Transformando.
treino_kbest = selecionar_k_melhores.transform( treino_x )
teste_kbest  = selecionar_k_melhores.transform( teste_x )

# Olhando shapes.
print( treino_kbest.shape )
print( teste_kbest.shape )

# Usando o Random Forest novamente.
classificador = RandomForestClassifier( n_estimators = 100, random_state = 1234 )
classificador.fit( treino_kbest, treino_y )
print( 'Resultado da classificação: %.2f%%' %( classificador.score( teste_kbest, teste_y ) * 100 ) )


    #                    # 
    # Matriz de Confusão #
    #                    # 

# Matriz de confusão.
matriz_confusao = confusion_matrix( teste_y, classificador.predict( teste_kbest ) )

# Exibindo.
matriz_confusao

# Mapa de calor do seaborn.
plt.figure( figsize = ( 10, 8 ) )
sns.heatmap( matriz_confusao, annot = True, fmt = 'd' ).set( xlabel = 'Predição', ylabel = 'Real' ) # mostrando valores e inteiros. 
plt.show()  


    #                                        #
    # Selecionando K melhores features (RFE) #
    #                                        #

# Fixando semente.
SEED = 1234
np.random.seed( SEED )

# Separando treino e teste.
treino_x, teste_x, treino_y, teste_y = train_test_split( valores_exames_v4, diagnostico, test_size = 0.3 )

# Usando o Random Forest novamente.
classificador = RandomForestClassifier( n_estimators = 100, random_state = 1234 )
classificador.fit( treino_x, treino_y )

# Selecionando usando RFE.
selecionar_rfe = RFE( estimator = classificador, n_features_to_select = 5, step = 1 )
selecionar_rfe.fit( treino_x, treino_y )
treino_rfe = selecionar_rfe.transform( treino_x )
teste_rfe  = selecionar_rfe.transform( teste_x )
classificador.fit( treino_rfe, treino_y )
matriz_confusao = confusion_matrix( teste_y, classificador.predict( teste_rfe ) )
plt.figure( figsize = ( 10, 8 ) )
sns.heatmap( matriz_confusao, annot = True, fmt = 'd' ).set( xlabel = 'Predição', ylabel = 'Real' ) # mostrando valores e inteiros. 
plt.show()  
print( 'Resultado da classificação: %.2f%%' %( classificador.score( teste_rfe, teste_y ) * 100 ) )


    #                                          #
    # Selecionando K melhores features (RFECV) #
    #                                          #

# Fixando semente.
SEED = 1234
np.random.seed( SEED )

# Separando treino e teste.
treino_x, teste_x, treino_y, teste_y = train_test_split( valores_exames_v4, diagnostico, test_size = 0.3 )

# Usando o Random Forest novamente.
classificador = RandomForestClassifier( n_estimators = 100, random_state = 1234 )
classificador.fit( treino_x, treino_y )

# Selecionando usando RFECV.
selecionar_rfecv = RFECV( estimator = classificador, cv = 5, step = 1, scoring = 'accuracy' ) # quantidade de CV (cross-validated) e método de avaliação (scoring).
selecionar_rfecv.fit( treino_x, treino_y )
treino_rfecv = selecionar_rfecv.transform( treino_x )
teste_rfecv  = selecionar_rfecv.transform( teste_x )
classificador.fit( treino_rfecv, treino_y )
matriz_confusao = confusion_matrix( teste_y, classificador.predict( teste_rfecv ) )
plt.figure( figsize = ( 10, 8 ) )
sns.heatmap( matriz_confusao, annot = True, fmt = 'd' ).set( xlabel = 'Predição', ylabel = 'Real' ) # mostrando valores e inteiros. 
plt.show()  
print( 'Resultado da classificação: %.2f%%' %( classificador.score( teste_rfecv, teste_y ) * 100 ) )

# Olhando features selecionadas e quais são.
selecionar_rfecv.n_features_ # quantidade.
treino_x.columns[ selecionar_rfecv.support_ ] # quais são as features utilizadas.


    #            #  
    # PCA e TSNE #
    #            #
    
# Usando PCA (ideal usar nos dados padronizados).
pca = PCA( n_components = 2 )
valores_exames_v5 = pca.fit_transform( valores_exames_v3 )    
    
# Graficamente.
plt.figure( figsize = ( 14, 8 ) )
sns.scatterplot( x = valores_exames_v5[ :, 0 ], y = valores_exames_v5[ :, 1 ], hue = diagnostico )    
plt.show()

# Usando TSNE (ideal usar nos dados padronizados).
tsne = TSNE( n_components = 2 ) # o TSNE gera uma maior distância entre os pontos originais.
valores_exames_v6 = tsne.fit_transform( valores_exames_v3 )    
    
# Graficamente.
plt.figure( figsize = ( 14, 8 ) )
sns.scatterplot( x = valores_exames_v6[ :, 0 ], y = valores_exames_v6[ :, 1 ], hue = diagnostico )    
plt.show()

