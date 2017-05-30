#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define SIGMOID(x) (1.0 / (1.0 + exp(-(x))));
#define L_RELU(x) ((x) >= 0 ? (x) : (x)/10.0);
#define MIN(x, y) ((x)<(y) ? (x) : (y));
#define MAX(x, y) ((x)>(y) ? (x) : (y));

typedef unsigned char BYTE;

typedef struct tagMNIST_DATA
{
	BYTE *pixels;
	BYTE label;
} MNIST, *pMNIST;

void forward_calculation(MNIST *data, double *computed_hidden, double *computed_output);
void weight_initialization(double *weight, int row, int column);
int argmax(double* array, int size);
void ann_testing();
void ann_training();
int little_to_big_endian(FILE *fd); 
MNIST* data_reader(MNIST *data_set, char *data_file, char *label_file, BYTE file_type); 

const double LEARNING_RATE = 0.1;
const int NUM_OF_INPUT = 28*28;
const int NUM_OF_HIDDEN = (28*28+10)/2;
const int NUM_OF_OUTPUT = 10;

int test_size;
int training_size;

int data_rows;
int data_columns;

double momentum = 0.2;	// momentum factor
/*
double gain;		// gain of sigmoid function
double error;		// total net error
*/

double *weight_hidden;
double *weight_output;

MNIST *training_data;
MNIST *test_data;

int print = 0;
int toggle = 0;
int epoches = 1;

/*
	feed forward calcuration
*/
void 
forward_calculation(MNIST *data, double *computed_hidden, double *computed_output)
{
	int i, j;
	
	for(i=1; i<=NUM_OF_HIDDEN; i++)
	{
		*(computed_hidden+i) = 0;

		for(j=1; j<=NUM_OF_INPUT; j++)
		{
			*(computed_hidden+i) += *(weight_hidden+NUM_OF_INPUT*i+j) 
					* ((double)data->pixels[j-1]/255.0);
		}

		if(toggle == 1)	printf("%lf\t", *(computed_hidden+i));
		*(computed_hidden+i) = SIGMOID(*(computed_hidden+i));
		if(toggle == 1)	printf("%lf\n", *(computed_hidden+i));
	}
	// Calculate Hidden
	
	if(toggle == 1)	printf("--------------------------------------\n");

	for(i=1; i<=NUM_OF_OUTPUT; i++)
	{
		*(computed_output+i) = 0;

		for(j=1; j<=NUM_OF_HIDDEN; j++)
		{
			*(computed_output+i) += *(weight_output+NUM_OF_HIDDEN*i+j) 
					* *(computed_hidden+j);
		}

		if(toggle == 1)	printf("%lf\t", *(computed_output+i));
		*(computed_output+i) = SIGMOID(*(computed_output+i));
		if(toggle == 1)	printf("%lf\n", *(computed_output+i));
	}
	// Calculate Output
}

/*
	Initializing weight using random value
*/
void
weight_initialization(double *weight, int row, int column) 
{
	int i,j;
	srand(time(NULL));

	for(i=0; i<=row; i++)
		for(j=1; j<=column; j++)
			*(weight+i*column+j) = 2*(((double)rand() / (double)RAND_MAX) - 0.5);
}

/*
	Get Index which make biggest value
*/
int
argmax(double* array, int size)
{
	int i;
	double max_value = array[0];
	int result = 0;

	for(i=1; i<=size; i++)
	{
		if(max_value < array[i])
		{
			max_value = array[i];
			result = i;
		}
	}
	return result-1;
}

/*
	Test Neural Network
*/
void
ann_testing()
{
        double *computed_hidden = (double *)malloc((NUM_OF_HIDDEN+1) * sizeof(double));
        double *computed_output = (double *)malloc((NUM_OF_OUTPUT+1) * sizeof(double));

        int correct = 0;
        int wrong = 0;

        int i, j;
        for(i=0; i<test_size; i++)
        {
                forward_calculation((test_data+i), computed_hidden, computed_output);

                int classification = argmax(computed_output, NUM_OF_OUTPUT);
		int label = (test_data+i)->label;

		if(classification == label)	correct++;
		else				wrong++;
	}

        printf("--Score--\n");
        printf("\tcorrect:\t%d/%d\n", correct, test_size);
        printf("\twrong:\t%d/%d\n", wrong, test_size);
        printf("\tAccuracy:\t%.3lf\n", (double)correct/(double)test_size);

        free(computed_hidden);
        free(computed_output);
}

/*
	Training Neural Network
*/
void 
ann_training() 
{
	weight_hidden = (double *)malloc((NUM_OF_INPUT+1) * (NUM_OF_HIDDEN+1) * sizeof(double));
	weight_output = (double *)malloc((NUM_OF_HIDDEN+1) * (NUM_OF_OUTPUT+1) * sizeof(double));

	double *d_weight_hidden = (double *)malloc((NUM_OF_INPUT+1) 
						* (NUM_OF_HIDDEN+1) * sizeof(double));
	double *d_weight_output = (double *)malloc((NUM_OF_HIDDEN+1) 
						* (NUM_OF_OUTPUT+1) * sizeof(double));

	double *error_hidden = (double *)malloc((NUM_OF_HIDDEN+1) * sizeof(double));
	double *error_output = (double *)malloc((NUM_OF_OUTPUT+1) * sizeof(double));
	
	double *computed_hidden = (double *)malloc((NUM_OF_HIDDEN+1) * sizeof(double));	
	double *computed_output = (double *)malloc((NUM_OF_OUTPUT+1) * sizeof(double));

	weight_initialization(weight_hidden, NUM_OF_HIDDEN, NUM_OF_INPUT);
	weight_initialization(weight_output, NUM_OF_OUTPUT, NUM_OF_HIDDEN);
	
	int i, j, k, l;

	for(i=0; i<NUM_OF_INPUT * NUM_OF_HIDDEN; i++)	*(d_weight_hidden) = 0;
	for(i=0; i<NUM_OF_HIDDEN * NUM_OF_OUTPUT; i++)	*(d_weight_output) = 0;
	
	for(i=0; i<epoches; i++)
	{
		printf("epoch: %d\n", i);
		
		for(j=0; j<training_size; j++)
		{
			int label = (training_data+j)->label;

			forward_calculation((training_data+j), computed_hidden, computed_output);
			// feed forward calculation			

			if(toggle == 1)	printf("target: %d", label);

			for(k=0; k<NUM_OF_OUTPUT; k++) 
			{
				*(error_output+k) = *(computed_output+k) 
							* (1.0 - *(computed_output+k)) 
							//* (label/10.0 - *(computed_output+k));
							* (label==i?1:0 - *(computed_output+k));
			}
			// delta k - Error of Output

			for(k=1; k<=NUM_OF_HIDDEN; k++)
			{
				double sum = 0.0;

				for(l=1; l<=NUM_OF_OUTPUT; l++)
					sum += *(weight_output+NUM_OF_OUTPUT*l+k) 
						* *(error_output+l);
 
				*(error_hidden+k) = *(computed_hidden+k) 
							* (1.0 - *(computed_hidden+k)) 
							* sum;
			}
			// delta h - Error of Hidden

			for(k=1; k<=NUM_OF_OUTPUT; k++) 
			{
				*(d_weight_output+k) = LEARNING_RATE 
						* error_hidden[k]
						+ momentum 
						* *(d_weight_output+k);
				*(weight_output+k) += *(d_weight_output+k);
			
				for(l=1; l<=NUM_OF_HIDDEN; l++) 
				{
					*(d_weight_output+NUM_OF_HIDDEN*k+l)
						= LEARNING_RATE
						* *(error_output+k)
						* *(computed_hidden)
						+ momentum
						* *(d_weight_output+NUM_OF_HIDDEN*k+l);

					*(weight_output+NUM_OF_HIDDEN*k+l)
						+= *(d_weight_output+NUM_OF_HIDDEN*k+l);
				}
			}
			// output layer weight update

			for(k=1; k<=NUM_OF_HIDDEN; k++) 
			{
				*(d_weight_hidden+k) = LEARNING_RATE 
						* error_hidden[k]
						+ momentum 
						* *(d_weight_hidden+k);
				*(weight_hidden+k) += *(d_weight_hidden+k);

				for(l=1; l<=NUM_OF_INPUT; l++)
				{
					*(d_weight_hidden+NUM_OF_INPUT*k+l) 
						= LEARNING_RATE 
						* error_hidden[k]
						* ((double)(training_data+j)->pixels[l-1]/255.0)
						+ momentum
						* *(d_weight_hidden+NUM_OF_INPUT*k+l);

					*(weight_hidden+NUM_OF_INPUT*k+l) 
						+= *(d_weight_hidden+NUM_OF_INPUT*k+l); 
				}
			}
			// hidden layer weight update
		
			if(j > 10000 && print == 1)	toggle = 1;	

			if(toggle == 1) 
			{
				int result = argmax(computed_output, NUM_OF_OUTPUT);
				printf("(%lf)\nresult: %d(%lf)\n", 
					computed_output[label+1], result, computed_output[result+1]);
				sleep(2);
			}

		}
	}
	
	free(error_hidden);
	free(error_output);

	free(computed_hidden);
	free(computed_output);

	free(d_weight_hidden);
	free(d_weight_output);
}

/*
	MNIST data follow little endian
	and Intel Processor follow big endian
	So this function change little endian to big endian
*/
int 
little_to_big_endian(FILE *fd) 
{
	BYTE *reader = (BYTE *)malloc(sizeof(BYTE)*4);
	fread(reader, sizeof(int), 1, fd);
	return (int)reader[3] | (int)reader[2]<<8 
				| (int)reader[1]<<16 | (int)reader[0]<<24;
}


/*
	Read MNIST Data File
*/
MNIST*
data_reader(MNIST *data_set, char *data_file, char *label_file, BYTE file_type) 
{
	FILE *f_data = fopen(data_file, "rb");
	FILE *f_label = fopen(label_file, "rb");

	if(!f_data || !f_label) 
	{
		printf("[Error code -1] %s file not exist!!\n", !f_data ? data_file : label_file);
		exit(-1);
	}

	int d_magic, l_magic;
	int num_of_data, num_of_label;
	
	d_magic = little_to_big_endian(f_data);
	l_magic = little_to_big_endian(f_label);

	if(d_magic != 2051 || l_magic != 2049)
	{
		printf("[Error code -2] This file isn't a MNIST file!\n");
		printf("\t\tMagic Number[data:%d, label:%d]\n", d_magic, l_magic);
		exit(-2);
	}

	num_of_data = little_to_big_endian(f_data);
	num_of_label = little_to_big_endian(f_label);

	if(num_of_data != num_of_label)
	{
		printf("[Error code -3] Number of data is not equal number of label\n");
		printf("\t\t Num of data: %d, Num of label: %d\n", num_of_data, num_of_label);
		exit(-3);
	}

	data_rows = little_to_big_endian(f_data);
	data_columns = little_to_big_endian(f_data);

	if(file_type == 0)	training_size = num_of_data;
	else			test_size = num_of_data;

	data_set = (MNIST *)malloc(sizeof(MNIST) * num_of_data);

	BYTE *data = (BYTE *)malloc(sizeof(BYTE));

	int i, j, k;
	for(i=0; i<num_of_data; i++)
	{
		fread(&((data_set+i)->label), sizeof(BYTE), 1, f_label);
	
		((data_set+i)->pixels) = (BYTE *)malloc(sizeof(BYTE) * data_rows * data_columns);
		for(j=0; j<data_rows; j++)
		{
			for(k=0; k<data_columns; k++)
			{
				fread(data, sizeof(BYTE), 1, f_data);
				(data_set+i)->pixels[j*data_columns + k] = *data;
			}
		}	
	}

	free(data);

	fclose(f_data);
	fclose(f_label);

	return data_set;
}

int 
main(int argc, char* argv) 
{
	char *tr_data = "MNIST/train-images.idx3-ubyte";
	char *tr_label = "MNIST/train-labels.idx1-ubyte";

	char *ts_data = "MNIST/t10k-images.idx3-ubyte";
	char *ts_label = "MNIST/t10k-labels.idx1-ubyte";

	training_data = data_reader(training_data, tr_data, tr_label, 0);
	test_data = data_reader(test_data, ts_data, ts_label, 1);
	printf("File Processing Complete!!\n");

	ann_training();
	printf("Training Complete!!\n");

	ann_testing();

	free(training_data);
	free(test_data);

	return 0;
}
