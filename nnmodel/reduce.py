
i = 0
with open('../embeddings/glove_s300.txt', 'r') as infile:
    with open('../embeddings/mini_glove_s300.txt', 'w') as outfile:
        for line in infile:
            outfile.write(line)
            i+=1
            if i > 10000:
                raise ValueError('done with this shit')   
