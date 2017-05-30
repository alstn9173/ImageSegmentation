package mnist;
import java.io.*;
import java.nio.file.*;
import java.util.*;

public class ANNClassification {
	private ArrayList<MNIST> trainingData;
	private ArrayList<MNIST> testData;

	private double[][] weightHidden;
	private double[][] weightOutput;
	
	private double[][] tempHidden;
	private double[][] tempOutput;

	private int rows;
	private int columns;

	private int dataSize;
	private int labelSize;

	public ANNClassification() {
		trainingData = new ArrayList<MNIST>();
		testData = new ArrayList<MNIST>();
	}

	public void saveWeight(int numOfHidden, int numOfOutput) {
		try {
			BufferedWriter fileWriter = new BufferedWriter(new FileWriter("weight.wt"));

			fileWriter.write("@tag mnist\n");

			fileWriter.write("<<StartOfHiddenLayer>>\n");
			for(int i=0; i<weightHidden.length; i++)
				for(int j=0; j<weightHidden[i].length; j++)
					fileWriter.write(weightHidden[i][j]+"\n");
			fileWriter.write("<<EndOfHiddenLayer>>\n");

			fileWriter.write("<<StartOfOutputLayer>>\n");
			for(int i=0; i<weightOutput.length; i++)
				for(int j=0; j<weightOutput[i].length; j++)
					fileWriter.write(weightOutput[i][j]+"\n");
			fileWriter.write("<<EndOfOutputLayer>>\n");

			System.out.println("Weight File Generated!!");

			fileWriter.close();
		}
		catch(Exception e) {
			System.err.println("write error!!");
			e.printStackTrace();
		}
	}

	private int negativeFixer(byte value) {
		return value >= 0 ? (int)value : (int)value+256;
	}

	private int endianTransform(byte n1, byte n2, byte n3, byte n4) {
		int a = negativeFixer(n1);
		int b = negativeFixer(n2);
		int c = negativeFixer(n3);
		int d = negativeFixer(n4);
		return a*256*256*256 + b*256*256 + c*256 + d;
	}

	private void fileReader(int numOfInput, int numOfHidden, int numOfOutput) {
		weightHidden = new double[numOfHidden][numOfInput];
		weightOutput = new double[numOfOutput][numOfHidden];

		try {
			Scanner file = new Scanner(new FileReader("weight.wt"));

			while(file.hasNextLine()) {
				String line = file.nextLine();

				if(line.equals("<StartOfHiddenLayer>>")) {
					while(line.equals("<<EndOfHiddenLayer>>")) {
						line = file.nextLine();
						for(int i=0; i<weightHidden.length; i++)
							for(int j=0; j<weightHidden[i].length; j++)
								weightHidden[i][j] = Double.parseDouble(line);

						System.out.println(line);
					}
				}
				else {
					if(line.equals("<StartOfHiddenLayer>>")) {
						while(line.equals("<<EndOfOutputLayer>>")) {
							line = file.nextLine();
							for(int i=0; i<weightOutput.length; i++)
								for(int j=0; j<weightOutput[i].length; j++)
									weightOutput[i][j] = Double.parseDouble(line);
						}
					}
				}
			}

			file.close();
		}
		catch (FileNotFoundException noFile) {
			System.err.println("File not found");
			// 지정된 파일이 존재하지 않을 경우
			System.exit(-1);
		}
		catch(InputMismatchException misMatch) {
			System.err.println("Data type is not correct!");
			// 들어온 데이터의 포맷(타입)이 맞지 않을 경우
			System.exit(-2);
		}
		catch(NoSuchElementException noData) {
			System.err.println("Missing data exist!!");
			// 들어온 데이터에 missing data가 존재할 경우
			System.exit(-3);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}

	private void dataReader(ArrayList<MNIST> storage, String data, String label)  {
		// 파일에 담긴 데이터를 읽어 저장하는 메소드
		try {
			/*
			 * [0-3] = magic  number
			 * [4-7] = data size
			 * [8-11] = data rows
			 * [12-15] = data columns
			 */
			Path dataPath = Paths.get(data);
			byte[] dataFile = Files.readAllBytes(dataPath);

			Path labelPath = Paths.get(label);
			byte[] labelFile = Files.readAllBytes(labelPath);

			int dataIndex = 0;
			int labelIndex = 0;

			int dataMagic = endianTransform(dataFile[dataIndex++], dataFile[dataIndex++], dataFile[dataIndex++], dataFile[dataIndex++]);
			int labelMagic = endianTransform(labelFile[labelIndex++], labelFile[labelIndex++], labelFile[labelIndex++], labelFile[labelIndex++]);

			dataSize = endianTransform(dataFile[dataIndex++], dataFile[dataIndex++], dataFile[dataIndex++], dataFile[dataIndex++]);
			labelSize = endianTransform(labelFile[labelIndex++], labelFile[labelIndex++], labelFile[labelIndex++], labelFile[labelIndex++]);

			if(dataMagic != 2051 || labelMagic != 2049 || (dataSize != labelSize))
				throw new InputMismatchException();

			rows = endianTransform(dataFile[dataIndex++], dataFile[dataIndex++], dataFile[dataIndex++], dataFile[dataIndex++]);
			columns = endianTransform(dataFile[dataIndex++], dataFile[dataIndex++], dataFile[dataIndex++], dataFile[dataIndex++]);

			for(int i=0; i<dataSize; i++) {
				MNIST temp = new MNIST(rows, columns);

				temp.setLabel(labelFile[labelIndex++]);
				for(int j=0; j<rows; j++)
					for(int k=0; k<columns; k++)
						temp.setPixel(negativeFixer(dataFile[dataIndex++]), j, k);

				storage.add(temp);
			}
		} 
		catch (FileNotFoundException noFile) {
			System.err.println("File not found");
			// 지정된 파일이 존재하지 않을 경우
			System.exit(-1);
		}
		catch(InputMismatchException misMatch) {
			System.err.println("Data type is not correct!");
			// 들어온 데이터의 포맷(타입)이 맞지 않을 경우
			System.exit(-2);
		}
		catch(NoSuchElementException noData) {
			System.err.println("Missing data exist!!");
			// 들어온 데이터에 missing data가 존재할 경우
			System.exit(-3);
		}
		catch(Exception e) {
			e.printStackTrace();
		}
	}
	
	private double[][] deepCopy(double[][] target, double[][] source) {
		for(int i=0; i<target.length; i++)
			for(int j=0; j<target[i].length; j++)
				target[i][j] = source[i][j];
		return target;
	}

	private void annTraining(double eta, int numOfInput, int numOfHidden, int numOfOutput) {
		weightHidden = new double[numOfHidden][numOfInput];
		weightOutput = new double[numOfOutput][numOfHidden];

		double[] errorOutput = new double[numOfOutput];
		double[] errorHidden = new double[numOfHidden];

		double[] computedOutput = new double[numOfOutput];		// Ok
		double[] computedHidden = new double[numOfHidden];		// Oh

		weightInitialize(weightHidden);
		weightInitialize(weightOutput);
		// initialize weight value using random value
		
		tempHidden = new double[numOfHidden][numOfInput];
		tempOutput = new double[numOfOutput][numOfHidden];
		
		tempHidden = deepCopy(tempHidden, weightHidden);
		tempOutput = deepCopy(tempOutput, weightOutput);

		long startTime = System.currentTimeMillis();

		double minError = Double.MAX_VALUE;
		double currError = 0;

		for(int l = 0 ; l < 1; l++) {
			//while(threshold < 0.8) {
			System.out.println("Epoches: " + l);

			int counter = 0;
			int percent = 0;
			for(MNIST data : trainingData) {
				int answer = data.getLabel();

				int size = trainingData.size();
				if(counter%(size/10)== 0) {
					System.out.println(percent + " %");
					percent+=10;
				}
				counter++;

				for(int j=0; j<numOfHidden; j++) {
					computedHidden[j] = 0;

					for(int i=0; i<numOfInput; i++)
						computedHidden[j] += weightHidden[j][i] * data.getPixel(i/columns, i%columns) / 255.0;

					computedHidden[j] = sigmoid(computedHidden[j]);
					//computedHidden[j] = softmax(computedHidden, j);

				}
				// input -> hidden layer feed forward calculation

				for(int j=0; j<numOfOutput; j++) {
					computedOutput[j] = 0;

					for(int i=0; i<numOfHidden; i++)
						computedOutput[j] += weightOutput[j][i] * computedHidden[i];

					computedOutput[j] = sigmoid(computedOutput[j]);
					//computedOutput[j] = softmax(computedOutput, j);
				}
				// hidden -> output layer feed forward calculation

				for(int i=0; i<numOfOutput; i++)
					errorOutput[i] = computedOutput[i] * (1 - computedOutput[i]) * (answer==i?1:0 - computedOutput[i]);
				// δk <- Ok * (1 - Ok) * (tk - Ok) 
				// output layer error calculation

				for(int i=0; i<numOfHidden; i++) {
					double sum = 0.0;

					for(int j=0; j<numOfOutput; j++)
						sum += weightOutput[j][i] * errorOutput[j];

					errorHidden[i] = computedHidden[i] * (1 - computedHidden[i]) * sum;
				}
				// δh <- Oh * (1 - Oh) * Sum(Wkh * δk) 
				// hidden layer error calculation

				for(int j=0; j<numOfOutput; j++)
					for(int i=0; i<numOfHidden; i++)
						weightOutput[j][i] += eta * errorOutput[j] * computedHidden[i];
				// output layer weight update

				for(int j=0; j<numOfHidden; j++)
					for(int i=0; i<numOfInput; i++)
						weightHidden[j][i] += eta * errorHidden[j] * data.getPixel(i/columns, i%columns) / 255.0;
				// wji <- wji + Δwji 
				// when Δwji = η * δj * xji
				// hidden layer weight uptdate

				//System.out.println("target: " + data.getLabel() + ", answer: " + argmax(computedOutput));

				currError = 0;
				for(int i=0; i<numOfOutput; i++) {
					if(i==answer)		currError += 1.0 - computedOutput[i];
					else					currError += computedOutput[i];
				}
				currError /= numOfOutput;
				// MSE
				
				if(currError < minError) {
					minError = currError;
					
					tempHidden = deepCopy(tempHidden, weightHidden);
					tempOutput = deepCopy(tempOutput, weightOutput);
				}
				else {
					//System.out.println("changed!! " + currError + "/" + minError);
					
					//weightHidden = deepCopy(weightHidden, tempHidden);
					//weightOutput = deepCopy(weightOutput, tempOutput);
				}
				// Error Feedback
			}
		}

		long endTime = System.currentTimeMillis();
		double elipsedTime = (endTime - startTime) / 1000.0;
		System.out.printf("Time Taken to training data: %.2f seconds\n", elipsedTime);

		System.out.printf("End phase Error : %.8f\n", currError);
		System.out.printf("Minimal Error : %.8f\n", minError);
	}

	private void weightInitialize(double[][] weight) {
		for(int i=0; i<weight.length; i++)
			for(int j=0; j<weight[i].length; j++)
				weight[i][j] = 2*(Math.random()-0.5);
	}

	public void printWeight(double[][] weight) {
		for(int i=0; i<weight.length; i++) {
			for(int j=0; j<weight[i].length; j++)
				System.out.print("["+i+","+j+"] = " + weight[i][j] + "\t");
			System.out.println();
		}
		System.out.println();
	}

	private void annTest(int numOfInput, int numOfHidden, int numOfOutput) {
		double[] computedOutput = new double[numOfOutput];		// Ok
		double[] computedHidden = new double[numOfHidden];		// Oh

		int[][] confusionMatrix = new int[numOfOutput][numOfOutput];

		int correct = 0;
		int wrong = 0;

		for(MNIST test: testData) {
			for(int j=0; j<numOfHidden; j++) {
				computedHidden[j] = 0;

				for(int i=0; i<numOfInput; i++)
					computedHidden[j] += weightHidden[j][i] * test.getPixel(i/columns, i%columns);

				computedHidden[j] = sigmoid(computedHidden[j]);
			}
			// input -> hidden layer feed forward calculation

			for(int j=0; j<numOfOutput; j++) {
				computedOutput[j] = 0;

				for(int i=0; i<numOfHidden; i++)
					computedOutput[j] += weightOutput[j][i] * computedHidden[i];

				computedOutput[j] = sigmoid(computedOutput[j]);
			}
			// hidden -> output layer feed forward calculation

			int result = argmax(computedOutput);

			if(result == test.getLabel())	correct++;
			else									wrong++;

			confusionMatrix[test.getLabel()][result]++;			
		}

		System.out.println("correct: " + correct);
		System.out.println("wrong: " + wrong);
		System.out.printf("Accuracy: %.3f\n\n", correct/(double)dataSize);

		for(int i=0; i<10; i++)
			System.out.printf("%4d ", i);
		System.out.println();

		for(int i=0; i<numOfOutput; i++) {
			for(int j=0; j<numOfOutput; j++) {
				System.out.printf("%4d ", confusionMatrix[i][j] );
			}
			System.out.println("\t" + i);
		}

	}

	private int argmax(double[] array) {
		int maxIndex = 0;
		double maxValue = array[0];
		for(int i=1; i<array.length; i++) {
			if(array[i] > maxValue) {
				maxIndex = i;
				maxValue = array[i];
			}
		}
		return maxIndex;
	}

	private double sigmoid(double expo) {	return 1.0/(1.0 + Math.exp(-expo));		}

	private double leakyReLU(double value) {	return value >= 0 ? value : value/10.0;	}

	private double softmax(double[] array, int k) {
		double exp = Math.exp(array[k]);
		double sumOfExp = 0;

		for(int i=0; i<array.length; i++)
			sumOfExp += Math.exp(array[i]);

		return exp/sumOfExp;
	}

	public void run() {
		int numOfInput = 28*28;
		int numOfHidden = (28*28+10)/2;
		int numOfOutput = 10;

		Scanner in = new Scanner(System.in);
		System.out.println("Default ANN --> anykey | ReadFile --> r");
		System.out.print(">> ");
		String command = in.nextLine();

		if(!command.equals("r")) {
			dataReader(trainingData, "./MNIST/train-images.idx3-ubyte", "./MNIST/train-labels.idx1-ubyte");
			dataReader(testData, "./MNIST/t10k-images.idx3-ubyte", "./MNIST/t10k-labels.idx1-ubyte");

			double etha = 0.1;
			numOfInput = rows*columns;
			numOfOutput = 10;
			numOfHidden = (numOfInput + numOfOutput)/2;

			annTraining(etha, numOfInput, numOfHidden, numOfOutput);
		}
		else {
			fileReader(numOfInput, numOfHidden, numOfOutput);

			printWeight(weightOutput);
		}

		annTest(numOfInput, numOfHidden, numOfOutput);
		
		System.out.println("--------------------------------------------");
		System.out.println("minimal error weight");
		weightHidden = deepCopy(weightHidden, tempHidden);
		weightOutput = deepCopy(weightOutput, tempOutput);
		
		annTest(numOfInput, numOfHidden, numOfOutput);

		if(!command.equals("r")) {
			System.out.println("Save Weight? (y/else)");
			System.out.print(">> ");
			command = in.nextLine();

			if(command.equals("y"))
				saveWeight(numOfInput * numOfHidden, numOfHidden * numOfOutput);
		}
		System.out.println("Program Terminated!!");

		in.close();
	}

	public static void main(String[] args) {
		ANNClassification wine = new ANNClassification();

		wine.run();

	}
}
