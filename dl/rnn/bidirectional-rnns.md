There have been many advancements with RNNs \([attention](https://www.oreilly.com/ideas/interpretability-via-attentional-and-memory-based-interfaces-using-tensorflow), Quasi RNNs, etc.\) that we will cover in later lessons but one of the basic and widely used ones are bidirectional RNNs \(Bi-RNNs\). The motivation behind bidirectional RNNs is to process an input sequence by both directions. Accounting for context from both sides can aid in performance when the entire input sequence is known at time of inference. A common application of Bi-RNNs is in translation where it's advantageous to look at an entire sentence from both sides when translating to another language \(ie. Japanese → English\).

![](/assets/runnbidirection1.png)



