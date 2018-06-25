# ShakespeareBot-5000
*CS 155 Machine Learning Data Mininig course project, Caltech, winter 2017. Equal contributor: Botao Hu, Jian Xu, Yukai Liu.*

## Delivery
Teach AI to Generate Shakespearean sonnets. Selected piece:

*__Hidden Markov Blues__   
Most wrongfully pains mow religious plot,  
Wherein entrap my heart are not grow sum!  
Lightening clear till never-resting blot,  
Exceeding saints outcast interchange some.  
Heretofore 'tis wait of would wastes old eyes,  
Corruption graced of witness appeal.  
Setting cloudy belong plain doth the wise,  
Flattery fair unworthy why ne'er feel?  
Columbines doth o, but breasts ere so hand,  
Devouring night meant 'Sorrows heat frost themes'.  
Injury despite dropping tempered stand,  
Benefit greater for did lilies beams.  
  External but in fulfil purer make,  
  Becoming do antique thee abstain take.*

## Approach
1. Pre-processing
* Tokenization: split the whole data set into different units such as words or phrases, and give each unit a unique index to distinguish them.
* Sequence: used each line as a sequence.
* Incorporated the poems of Shakespeare and Spenser as one large training data set.
* Generated two specialized dictionaries to map syllables and rhymes of each word that appears in the data set.
2. HMM
* Regularized unsupervised learning using Hidden Markov Models. Cross-validation applied.
* Naive version: poem-as-sequence HMM, didn't make much sense.
* Advanced version: line-as-sequence HMM, in rhyme, good syllabus status, logical and meaningful.
3. RNN
* GRU used through PyTorch.
* Naive character-based RNN: overfitting issues.
* Advanced techniques: more matrices applied, and tried word-based or word-embedding model.
