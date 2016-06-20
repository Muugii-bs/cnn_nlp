import sys

FILE_IN = sys.argv[1]
FILE_POS = sys.argv[2]
FILE_NEG = sys.argv[3]
FLAG = sys.argv[4]
CHUNK_SIZE = 56
MAX_SIZE = 11500

def main():
    if FLAG == 'fragile':
        fragile()
    elif FLAG == 'whole':
        passage()

def passage():
    with open(FILE_IN, 'r') as fp1:
        for line in fp1:
            tmp = line.split('\t')
            count = len(tmp[1].split())
            if tmp[0] == '0' and count < MAX_SIZE:
                with open(FILE_NEG, 'a') as fp2:
                    fp2.write(tmp[1])
            elif tmp[0] == '1' and count < MAX_SIZE:
                with open(FILE_POS, 'a') as fp3:
                    fp3.write(tmp[1])

def fragile():
    with open(FILE_IN, 'r') as fp1:
        for line in fp1:
            tmp = line.split('\t')
            words = tmp[1].rstrip().split()
            if tmp[0] == '0':
                with open(FILE_NEG, 'a') as fp2:
                    for i in range(0, len(words), CHUNK_SIZE):
                        fp2.write(' '.join(words[i:i+CHUNK_SIZE]) + '\n')
            elif tmp[0] == '1':
                with open(FILE_POS, 'a') as fp3:
                    for i in range(0, len(words), CHUNK_SIZE):
                        fp3.write(' '.join(words[i:i+CHUNK_SIZE]) + '\n')

if __name__ == '__main__':
    main()
