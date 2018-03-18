
# ## Stateful model data setup
#realistic last 20% of rows as validation

PATH='/mnt/samsung128/Data/nietzsche/data/'

TRN_PATH = 'trn/'
VAL_PATH = 'val/'
TRN = f'{PATH}{TRN_PATH}'
VAL = f'{PATH}{VAL_PATH}'

num_lines = sum(1 for line in open(PATH+'nietzsche.txt'))
trn_length = int(num_lines*0.8)
val_length = num_lines-trn_length

val_result = []
trn_result = []
with open(PATH+'nietzsche.txt', 'r') as fin:
    for line_number, line in enumerate(fin):
        if line_number > trn_length:  # line_number starts at 0.
            val_result.append(line)
        else:
            trn_result.append(line)

#joining with /n created too many gaps so leaving out
with open(PATH+TRN_PATH+'training.txt', 'w') as ftrn:
    ftrn.write(''.join(trn_result))

with open(PATH+VAL_PATH+'validation.txt', 'w') as fval:
    fval.write(''.join(val_result))