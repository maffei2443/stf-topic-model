# Topic Modelling Brazilian Supreme Court Lawsuits

This repo holds the source code for the work described in the paper below:

* [Pedro H. Luz de Araujo](http://lattes.cnpq.br/8374005378743328) and [Teófilo E. de Campos](http://www.cic.unb.br/~teodecampos/)  
[Topic Modelling Brazilian Supreme Court Lawsuits](http://ebooks.iospress.nl/volumearticle/56168)  
[33rd International Conference on Legal Knowledge and Information Systems​ (JURIX 2020), held online in December 9-11, 2020](https://jurix2020.law.muni.cz/).

```
@InProceedings{luz_etal_jurix2020,
          author = {Pedro H. {Luz de Araujo} and Te\'{o}filo E. {de Campos}},
          title = {Topic Modelling Brazilian Supreme Court Lawsuits},
          booktitle = {International Conference on Legal Knowledge and Information Systems​ ({JURIX})},
	  publisher = {IOS Press},
	  series = {Frontiers in Artificial Intelligence and Applications},
	  pages = {113--122},
          year = {2020},
          month = {December 9-11},
          address = {Prague, Czech Republic},	  
	  doi = {10.3233/FAIA200855},
	  url = {http://ebooks.iospress.nl/volumearticle/56168},
}	  
```


The sections below describe the requirements and the files.

We kindly request that users cite our paper in any publication that is generated as a result of the use of our work.

## Requirements
1. [Python 3.6](https://www.python.org/downloads/)	
2. [Gensim](https://pypi.org/project/gensim/)
3. [pyLDAvis](https://pypi.org/project/pyLDAvis/)
4. [Pandas](https://pandas.pydata.org/)
5. [Scikir-learn](https://scikit-learn.org/stable/)
6. [XGBoost](https://xgboost.readthedocs.io/en/latest/)

## Files
* lda_explore.ipynb: notebook for topic model training
* train_xgboost_tfidf.ipynb: notebook for classifier training using tfidf values or word counts.
* train_xgboost_topics.ipynb: notebook for classifier training using topic distribution.
