# Natural-Language-Processing-Semantic-Role-Labeling
Semantic Role Labeling using BERT and BiLSTM

Semantic Role Labeling (SRL) is a Natural Language Processing (NLP) task in which we are interested in analyzing the predicate-argument structures within a sentence, or more intuitively answering to question of *”Who did What to Whom, Where, When, and How?” (Màrquez et al., 2008)*. A predicate is a word or a multi-word expression denoting an event or an action, and an argument is a part of the text linked in some way to a predicate. Once extracted the predicate-argument structures we perform two disambiguation steps by assigning to each predicate a sense and to each argument a semantic role. SRL can be seen as a pipeline of four subtasks: 
1. Predicate identification find predicates 
2. Predicate disambiguation assign a sense to each predicate 
3. Argument identification find the argument of each predicate 
4. Argument classification assign a semantic role to each argument.
In this paper we will perform SRL in two different scenarios: 
- SRL_34 results of predicate identification and disambiguation are given, therefore we serve only steps 3 and 4
- SRL_234 only predicate identification is provided, so tasks 2, 3, and 4 must be performed.

More information and the results in `report.pdf`.