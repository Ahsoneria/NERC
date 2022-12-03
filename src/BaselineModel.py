import datasets
from datasets import load_dataset

def percentage(total, misses):
    if total ==0:return 0
    else: return 1-float(misses/total)

def f1(prec, rec):
    return 2 * (prec * rec) / (prec + rec)

def getBaselineResults(dataset, name):
    print(name,':')
    i = 0
    totalTokens=0
    totalPositivePredictions=0
    totalFalsePositive=0
    totalFalseNegative=0
    tagCounts = {0:0,1:0,2:0,3:0,4:0,5:0,6:0,7:0,8:0}
    ner_tags = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    inv_tags = {v: k for k, v in ner_tags.items()}
    for d in dataset:
        #print(d)
        #print(len(d['tokens']), len(d['tokens']), len(d['pos_tags']), len(d['chunk_tags']), len(d['ner_tags']))
        i = i+1
        for token, ner, pos in zip(d['tokens'], d['ner_tags'], d['pos_tags']):
            totalTokens=totalTokens+1
            if pos==22:
                totalPositivePredictions=totalPositivePredictions+1
                if inv_tags[ner] =='O': totalFalsePositive = totalFalsePositive+1
                tagCounts[ner] = tagCounts[ner]+1
            elif inv_tags[ner] !='O': totalFalseNegative= totalFalseNegative+1

    totalTruePositive = totalPositivePredictions - totalFalsePositive

    #recognizedTotalPositives = totalPositivePredictions-totalFalsePositive+
    recognizedPrecision = totalTruePositive/(totalTruePositive+totalFalsePositive)
    recognizedRecall = totalTruePositive/(totalTruePositive + totalFalseNegative)
    print("Recognized: precision =",recognizedPrecision,"recall =",recognizedRecall,"f1 =", f1(recognizedPrecision,recognizedRecall))
    print(tagCounts)
    #lab=labeled
    labTruePositive=tagCounts[5]
    labFalsePositives=totalPositivePredictions-labTruePositive
    labeledPrecision = labTruePositive/(totalPositivePredictions)
    labeledRecall = labTruePositive/(labTruePositive + totalFalseNegative)
    print("Labeled: precision =",labeledPrecision,"recall =",labeledRecall, "f1 =", f1(labeledPrecision,labeledRecall))
    return recognizedPrecision, recognizedRecall, f1(recognizedPrecision,recognizedRecall), labeledPrecision, labeledRecall, f1(labeledPrecision,labeledRecall)

def getBaselinePredictions(sentences):
    predictions = []
    for s in sentences:
        #print(d)
        #print(len(d['tokens']), len(d['tokens']), len(d['pos_tags']), len(d['chunk_tags']), len(d['ner_tags']))
        for token, pos in zip(s['tokens'], s['pos_tags']):
            if pos == 22:
                predictions.append((token, 5))
            else: predictions.append((token,0))
    return predictions



dataset = load_dataset('conll2003', split='train')
getBaselineResults(dataset, 'train')
print('\n\n')
dataset = load_dataset('conll2003', split='test')
getBaselineResults(dataset, 'train')

sentences = []
sentences.append({'tokens': ['the', 'person', 'worked', 'at', 'Google'], 'pos_tags':[0,0,0,0,22]})
print(getBaselinePredictions(sentences))
