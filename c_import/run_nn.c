#include <stdio.h>

int main()
{
    FILE *file;
    unsigned char embedding[39][4][2];
    unsigned char rnn_w[4][32][2];
    unsigned char rnn_r[32][32][2];
    unsigned char rnn_b[1][32][2];
    unsigned char dense_w[32][1][2];
    unsigned char dense_b[1][1][2];

file = fopen("rnn_binary_model_0312_rnn32.bin", "r");


for(int i = 0, i < 39, i++){
    for(int j = 0, j < 4, j++){
        fread(embedding[i][j], 2, 1, file);
    }
}

for(int i = 0, i < 4, i++){
    for(int j = 0, j < 32, j++){
        fread(rnn_w[i][j], 2, 1, file);
    }
}

for(int i = 0, i < 32, i++){
    for(int j = 0, j < 32, j++){
        fread(rnn_r[i][j], 2, 1, file);
    }
}
    
for(int i = 0, i < 1, i++){
    for(int j = 0, j < 32, j++){
        fread(rnn_b[i][j], 2, 1, file);
    }
}
    
for(int i = 0, i < 32, i++){
    for(int j = 0, j < 1, j++){
        fread(dense_w[i][j], 2, 1, file);
    }
}

fread(dense_b[1][1], 2, 1, file);
    
fclose(file);

    return 0;
}
