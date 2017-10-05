package mlp.logic;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.Random;

public class MLP {
	int NI; //(number of inputs) 
	int NH; //(number of hidden units) 
	int NO; //(number of outputs) 
	double W1[][]; //(array containing the weights in the lower layer) 
	double W2[][]; //(array containing the weights in the upper layer) 
	double dW1[][];
	double dW2[][];//(arrays containing the weight *changes* to be applied onto W1 and W2) 
	double Z1[]; //(array containing the derivatives of functions for lower layer) 
	double Z2[]; //(array containing the derivatives of functions for the upper layer) 
	double H[]; //(array where the values of the hidden neurons are stored – need these saved to compute dW2) 
	double O[]; //(array where the outputs are stored) 

	public static void main(String[] args) {
		MLP nn=new MLP();
		nn.trainXOR();   //call to XOR trainer
		nn.trainSin();   // call to sin trainer
		nn.letterRecognition();   //call to UCI dataset letter recognition trainer
	} 

	void letterRecognition()
	{
		String line;
		try {
			InputStream fis = new FileInputStream("letter_recognition.data");
			InputStreamReader isr = new InputStreamReader(fis, Charset.forName("UTF-8"));
			BufferedReader br = new BufferedReader(isr);

			ArrayList<Object>targets =new ArrayList<Object>();
			ArrayList<Object>inputs =new ArrayList<Object>();

			while ((line = br.readLine()) != null) {

				ArrayList<Double>input =new ArrayList<Double>();
				ArrayList<Double>target =new ArrayList<Double>();

				String letter= line.split(",")[0]; 
				int index=letter.toCharArray()[0]-65;   //each letter is binary vector with 1 at its corresponding index. Rest positions are all zero 
				for(int i=0;i<26;i++)    // for e.g. A => {1,0,0,0,0,0,......0,0}
				{
					if(i==index)
					{
						target.add(1.0);  
					}
					else
						target.add(0.0); 
				}
				targets.add(target);
				for(int i=1;i<17;i++)
				{
					input.add(Double.parseDouble(line.split(",")[i]));    // add attributes input vector 
				}
				input.add(1.0); //bias
				inputs.add(input);
			}
			trainLetterRecog(inputs, targets);
		}
		catch(Exception e)
		{
			e.printStackTrace();
		}
	}

	void trainSin()
	{
		NI=4; //(number of inputs) 
		NH=10; //(number of hidden units) 
		NO=1; //(number of outputs) 
		setParams();
		Random r=new Random();
		ArrayList<Object>targets =new ArrayList<Object>();
		ArrayList<Object>inputs =new ArrayList<Object>();
		for(int i=0;i<50;i++)
		{
			ArrayList<Double>input =new ArrayList<Double>();
			double sum=0;
			for(int j=0;j<4;j++)
			{
				double x=(r.nextDouble()*2)-1;   //generate 50 vectors
				sum=sum+x;
				input.add(x);
			}
			input.add(1.0);  //bias
			inputs.add(input);
			ArrayList<Double>target =new ArrayList<Double>();
			target.add(Math.sin(sum));
			targets.add(target);
		}
		randomize();
		int e=0;
		int flag=0;
		int maxEpochs=90000;
		int numExamples=40;
		for (e=0; e<maxEpochs; e++) 
		{  
			flag=0;
			double error = 0;  
			for (int p=0; p< numExamples; p++) 
			{   
				forwardSin((ArrayList<Double>)inputs.get(p));
				double t=backwardSin((ArrayList<Double>)targets.get(p),(ArrayList<Double>)inputs.get(p),0.0099,flag);
				flag=1;
				int c=p+1;
				error = error+t; 

			} 
			System.out.println( "Error at epoch " + e + " is " + error +"\n"); 	
		}

		double test_error=errorOnTest(inputs, 40, 50, targets);
		System.out.println("Error on test set for sine function:"+test_error);
	}
	void setParams()   //method to set allocate memory to different arrays
	{
		W1=new double[NI+1][NH+1];    //+1 represents bias
		W2=new double[NH+1][NO]; 
		dW1=new double[NI+1][NH+1];
		dW2=new double[NH+1][NO]; 
		Z1=new double[NH+1]; 
		Z2=new double[NO]; 
		H=new double[NH+1]; 
		O=new double[NO]; 
	}
	void trainXOR()
	{
		NI=2; //(number of inputs) 
		NH=2; //(number of hidden units) 
		NO=1; //(number of outputs)  
		setParams();
		int maxEpochs=100000;

		//XOR
		int numExamples=4;

		ArrayList<Object>inputs =new ArrayList<Object>();
		ArrayList<Double>input =new ArrayList<Double>();
		input.add(0.0);
		input.add(0.0);
		input.add(1.0);  //bias
		inputs.add(input);
		input =new ArrayList<Double>();
		input.add(0.0);
		input.add(1.0);
		input.add(1.0);
		inputs.add(input);
		input =new ArrayList<Double>();
		input.add(1.0);
		input.add(0.0);
		input.add(1.0);
		inputs.add(input);
		input =new ArrayList<Double>();
		input.add(1.0);
		input.add(1.0);
		input.add(1.0);
		inputs.add(input);
		ArrayList<Object>targets =new ArrayList<Object>();
		ArrayList<Double>target =new ArrayList<Double>();
		target.add(0.0);
		targets.add(target);
		target=new ArrayList<Double>();
		target.add(1.0);
		targets.add(target);
		target=new ArrayList<Double>();
		target.add(1.0);
		targets.add(target);
		target=new ArrayList<Double>();
		target.add(0.0);
		targets.add(target);
		randomize();
		int e=0;
		int flag=0;
		for (e=0; e<maxEpochs; e++) 
		{  
			flag=0;
			double error = 0;  
			for (int p=0; p< numExamples; p++) 
			{   
				forwardXOR((ArrayList<Double>)inputs.get(p));
				double t=backwardXOR((ArrayList<Double>)targets.get(p),(ArrayList<Double>)inputs.get(p),0.1,flag);
				flag=1;
				int c=p+1;
				error = error+t; 

			} 
			System.out.println( "Error at epoch " + e + " is " + error +"\n"); 	
		}

		predict(inputs,targets);
	}
	void predict(ArrayList<Object>inputs,ArrayList<Object>targets)   //predict trained XOR neural net 
	{
		int numExamples=inputs.size();
		for (int p=0; p< numExamples; p++) 
		{
			forwardXOR((ArrayList<Double>)inputs.get(p));
			System.out.println("Target:"+((ArrayList<Double>)targets.get(p)).get(0));
			System.out.println("Predicted:"+O[0]);
		}
	}

	//method to find error for sin function on test data set
	double errorOnTest(ArrayList<Object>inputs,int start,int end,ArrayList<Object>targets)
	{
		double error=0;
		for (int p=start; p< end; p++) 
		{
			forwardSin((ArrayList<Double>)inputs.get(p));
			double t=((ArrayList<Double>)targets.get(p)).get(0);
			error=error+(0.5*(Math.pow((t-O[0]), 2)));
		}
		return error;
	}

	void randomize()  //method to randomize weights for XOR and sin functions
	{
		Random r=new Random();
		for(int i=0;i<NI+1;i++)
		{
			for(int j=0;j<NH+1;j++)
			{
				W1[i][j] = r.nextDouble();
				dW1[i][j]=0;
			}
		}
		for(int i=0;i<NH+1;i++)
		{
			for(int j=0;j<NO;j++)
			{
				W2[i][j] = r.nextDouble();
				dW2[i][j]=0;
			}
		}
	}
	void randomizeLetterRecog()   //assigning very very low initial weights for letter recognizer Neural network 
	{
		double upper1 = 1E-2;
		double lower2 = -8E-4;
		double upper11 = 1E-1;
		double lower22 = 2E-6;
		Random r=new Random();
		for(int i=0;i<NI+1;i++)
		{
			for(int j=0;j<NH+1;j++)
			{
				W1[i][j] = Math.random() * (upper1 - lower2) + lower2;
				dW1[i][j]=0;
			}
		}
		for(int i=0;i<NH+1;i++)
		{
			for(int j=0;j<NO;j++)
			{
				W2[i][j] = Math.random() * (upper11 - lower22) + lower22;
				dW2[i][j]=0;
			}
		}
	}
	void forwardXOR(ArrayList<Double> input)  
	{
		for(int j=0;j<NH+1;j++)   //hidden layer
		{
			H[j]=0.0;	
			for(int i=0;i<NI+1;i++)
			{
				H[j]=H[j]+(input.get(i)*W1[i][j]);	
			}
			H[j]=1.0/(1.0+(Math.exp(-H[j])));    //sigmoid for input layer
		}
		H[NH]=1; //bias

		for(int j=0;j<NO;j++)   //output layer
		{
			O[j]=0.0;	
			for(int i=0;i<NH+1;i++)
			{
				O[j]=O[j]+(H[i]*W2[i][j]);	
			}
			O[j]=1.0/(1.0+(Math.exp(-O[j])));   //sigmoid for output layer
		}
	}

	double backwardXOR(ArrayList<Double> target,ArrayList<Double> input, double lr,int flag)  //sigmoidal output layer
	{
		double error=0;
		for(int i=0;i<NO;i++)   //output layer
		{
			Z2[i]=(target.get(i)-O[i])*O[i]*(1-O[i]);  //delta for O/P to hidden layer
		}
		for(int i=0;i<NH+1;i++)   
		{
			for(int j=0;j<NO;j++)   
			{
				dW2[i][j]=dW2[i][j]+(Z2[j] * H[i]);
				if(flag==0)
				{
					W2[i][j]=W2[i][j] + lr* dW2[i][j];  //update weights after every epoch
					dW2[i][j]=0;
				}
			}
		}
		for(int i=0;i<NH+1;i++)   //hidden layer
		{
			double sum=0;
			for(int j=0;j<NO;j++)   
			{
				sum=sum + (W2[i][j]*Z2[j]);
			}
			Z1[i]=sum * H[i]*(1-H[i]);  //delta for hidden layer to input layer
		}
		for(int i=0;i<NI+1;i++)   
		{
			for(int j=0;j<NH+1;j++)   
			{
				dW1[i][j]=dW1[i][j]+(Z1[j] * input.get(i));
				if(flag==0)    //update weights after every epoch
				{
					W1[i][j]=W1[i][j] +lr*dW1[i][j];
					dW1[i][j]=0;
				}
			}
		}
		for(int j=0;j<NO;j++)   
		{
			error=error+(0.5*(Math.pow((target.get(j)-O[j]), 2)));   //least square loss function
		}
		return error;
	}

	void forwardSin(ArrayList<Double> input)  //linear function in output layer
	{
		for(int j=0;j<NH+1;j++)   //hidden layer
		{
			H[j]=0.0;	
			for(int i=0;i<NI+1;i++)
			{
				H[j]=H[j]+(input.get(i)*W1[i][j]);	
			}
			H[j]=1.0/(1.0+(Math.exp(-H[j])));  //sigmoid hidden layer
		}
		H[NH]=1; //bias  last neuron is for bias

		for(int j=0;j<NO;j++)   //output layer
		{
			O[j]=0.0;	
			for(int i=0;i<NH+1;i++)
			{
				O[j]=O[j]+(H[i]*W2[i][j]);	//linear output
			}
		}
	}

	double backwardSin(ArrayList<Double> target,ArrayList<Double> input, double lr,int flag)
	{
		double error=0;
		for(int i=0;i<NO;i++)   //output layer
		{
			Z2[i]=(target.get(i)-O[i]);   //delta
		}
		for(int i=0;i<NH+1;i++)   
		{
			for(int j=0;j<NO;j++)   
			{
				dW2[i][j]=dW2[i][j]+(Z2[j] * H[i]);
				if(flag==0)
				{
					W2[i][j]=W2[i][j] + lr* dW2[i][j];  //update weights after every epoch
					dW2[i][j]=0;
				}
			}
		}
		for(int i=0;i<NH+1;i++)   //hidden layer
		{
			double sum=0;
			for(int j=0;j<NO;j++)   
			{
				sum=sum + (W2[i][j]*Z2[j]);
			}
			Z1[i]=sum * H[i]*(1-H[i]);
		}
		for(int i=0;i<NI+1;i++)   
		{
			for(int j=0;j<NH+1;j++)   
			{
				dW1[i][j]=dW1[i][j]+(Z1[j] * input.get(i)); //update weights after every epoch
				if(flag==0)
				{
					W1[i][j]=W1[i][j] +lr*dW1[i][j];
					dW1[i][j]=0;
				}
			}
		}
		for(int j=0;j<NO;j++)   
		{
			error=error+(0.5*(Math.pow((target.get(j)-O[j]), 2)));  //least square loss function
		}
		return error;
	}

	void trainLetterRecog(ArrayList<Object>inputs,ArrayList<Object>targets)
	{
		NI=16; //(number of inputs) 
		NH=11; //(number of hidden units) 
		NO=26; //(number of outputs) 
		setParams();
		randomizeLetterRecog();
		int e=0;
		int flag=0;
		int maxEpochs=10000;
		double lr=0.0000075;   
		double training_limit=inputs.size()*0.8;   //partition dataset 4/5 part for training and 1/5 for test
		int train_limit=(int)training_limit;
		for (e=0; e<maxEpochs; e++) 
		{  

			flag=0;
			double error = 0; 
			for (int p=0; p< train_limit; p++) 
			{   
				forwardLetterRecog((ArrayList<Double>)inputs.get(p));
				double t=backwardLetterRecog((ArrayList<Double>)targets.get(p),(ArrayList<Double>)inputs.get(p),lr,flag);
				flag=1;
				error = error+t;
			} 	

			System.out.println( "Error at epoch " + e + " is " + error/train_limit +"\n"); 
		}
		letterRecogClassificationRate(targets,inputs,train_limit);
	}
	void forwardLetterRecog(ArrayList<Double> input)  //softmax function in output layer
	{
		for(int j=0;j<NH+1;j++)   //hidden layer
		{
			H[j]=0.0;	
			for(int i=0;i<NI+1;i++)
			{
				H[j]=H[j]+(input.get(i)*W1[i][j]);	
			}
			H[j]=1.0/(1.0+(Math.exp(-H[j])));      //sigmoidal hidden layer
		}
		H[NH]=1; //bias

		for(int j=0;j<NO;j++)   //output layer
		{
			O[j]=0.0;	
			for(int i=0;i<NH+1;i++)
			{
				O[j]=O[j]+(H[i]*W2[i][j]);	
			}
		}
		double denom=0;
		for(int j=0;j<NO;j++)   //output layer
		{
			denom=denom+ Math.exp(O[j]);
		}
		for(int j=0;j<NO;j++)   //output layer
		{
			O[j]=(Math.exp(O[j]))/denom;     //softmax ouput
		}
	}

	double backwardLetterRecog(ArrayList<Double> target,ArrayList<Double> input, double lr,int flag)
	{
		double error=0;
		for(int i=0;i<NO;i++)   //output layer
		{
			Z2[i]=-(O[i]-target.get(i));   //delta
		}
		for(int i=0;i<NH+1;i++)   
		{
			for(int j=0;j<NO;j++)   
			{
				dW2[i][j]=dW2[i][j]+(Z2[j] * H[i]);
				if(flag==0)
				{
					W2[i][j]=W2[i][j] + lr* dW2[i][j];  //update weight after every epoch
					dW2[i][j]=0;
				}

			}
		}
		for(int i=0;i<NH+1;i++)   //hidden layer
		{
			double sum=0;
			for(int j=0;j<NO;j++)   
			{
				sum=sum + (W2[i][j]*Z2[j]);
			}
			Z1[i]=sum * H[i]*(1-H[i]);
		}
		for(int i=0;i<NI+1;i++)   
		{
			for(int j=0;j<NH+1;j++)   
			{

				dW1[i][j]=dW1[i][j]+(Z1[j] * input.get(i));
				if(flag==0)
				{
					W1[i][j]=W1[i][j] +lr*dW1[i][j]; //update weight after every epoch
					dW1[i][j]=0;
				}

			}
		}
		for(int j=0;j<NO;j++)   
		{   // cross entropy loss function
			double e=-((target.get(j)*Math.log(O[j]))+((1-target.get(j))*Math.log(1-O[j])));
			error=error+e;
		}
		return error;
	}

	//method to find classification rate for letter recognition neural network
	void letterRecogClassificationRate(ArrayList<Object>targets,ArrayList<Object>inputs,int start)
	{
		double class_rate=0;
		char letters[]={'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'};
		for(int p=start;p<inputs.size();p++)
		{
			int letterindex=0;
			forwardLetterRecog((ArrayList<Double>)inputs.get(p));
			int maxIndex=0;
			for (int i = 0; i < 26; i++) {
				double prob = O[i];
				if ((prob > O[maxIndex])) {    // find highest probability in the vector of predicted output
					maxIndex = i;
				}
				if(((ArrayList<Double>)targets.get(p)).get(i)==1.0)
				{
					letterindex =i;   // find the target 
				}
			}
			if(letterindex==maxIndex)   //if both match increase correct classification counter
				class_rate++;
		}
		System.out.println("Letter recognition classification rate:"+ class_rate/(inputs.size()-start));
	}
}
