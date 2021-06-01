import torch
from char_map import TextTransform

def GreedyDecoder(output, labels, label_lengths, blank_label=28, collapse_repeated=True):


    arg_maxes = torch.argmax(output, dim=2)
    text_transform = TextTransform()
    decodes=[]
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(text_transform.int_to_text(labels[i][:label_lengths[i]].tolist()))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(text_transform.int_to_text(decode))
    return decodes, targets