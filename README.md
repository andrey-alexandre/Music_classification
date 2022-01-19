# Classificação do gênero musical

Neste projeto, foi realizada a unificação dos dados, antes separados por gênero, em um único arquivo contendo todas as informações necessárias para as etapas seguintes.

Durante a etapa de limpeza dos dados foi adotada a lematização dos documentos, remoção de espaços e todas as letras foram colocadas em minúsculo para facilitar a descoberta de palavras repetidas. Também foram removidos as pontuações repetidas, como traços e pontos finais duplos, uma vez que não apresentam interpretabilidade para o conjunto de dados. 

Os simbolos de repetição de uma estrutura da música (Frase ou estrofe) foram removidos, mas em um estudo mais longo, analisar sua importância, seja repetindo as estruturas, ou somente mantendo o simbolo. Todas as músicas foram mantidas, mesmo as em lingua estrangeira. Portanto, um estudo mais aprofundado permitirá analisar o impacto destas letras no modelo e avaliar, junto ao cliente, a necessidade de ser utilizado modelos multilingual.

Foram feitas análises relacionadas ao número de palavras e frases presentes em cada gênero musical, assim como testes estatísticos de diferença de múltiplos grupos (Teste de Kruskal Wallis) e testes ad-hoc de diferença de dois grupos (Teste U de Mann-Whitney). Foi percebido que todos os grupos são diferentes entre si, o que indica que estas duas variáveis poderiam ser utilizadas no modelo proposto.

Avaliando o prazo proposto e as limitações computacionais existentes, foram utilizados modelos mais tradicionais, como a Regressão Logística, SVM e árvores de decisão. Para permitir uma avaliação mais clara da capacidade de generalização dos modelos, foi utilizada a técnica de validação cruzada com 10 pastas, um número razoável para a limitação computacional, mas também para o número de amostras (3200 observações).

Tendo em vista que os dados são totalmente balanceados, com 800 registros para cada um dos gêneros, foram utilizadas as medidas mais simples, Acurácia e Sensibilidade. Desta forma, o modelo selecionado foi a Regressão Logística, sendo que o modelo SVM com Kernel RBF apresentou resultados próximos, porém inferiores.

Portanto, foi salvo um objeto no formato joblib na pasta 'Docs' para que o modelo possa ser consumido por um sistema que apresentará os resultados para o cliente.
