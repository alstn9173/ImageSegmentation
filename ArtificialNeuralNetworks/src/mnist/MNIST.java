package mnist;

public class MNIST {
	private int label;
	private int[][] pixels;
	
	public MNIST(int row, int columns) {
		label = -1;
		pixels = new int[row][columns];
	}
	
	public void setLabel(int label) { this.label = label; }
	public void setPixel(int value, int row, int col) { pixels[row][col] = value; }
	
	public int getLabel() { return label; }
	public int getPixel(int i, int j) { return pixels[i][j]; }
}
