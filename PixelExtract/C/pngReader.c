#include <stdio.h>
#include <stdlib.h>
#include <png.h>

int image_width, image_height;
png_byte color_type;
png_byte bit_depth;
png_bytep *row_pointers;

/**
	@TODO	
	- read sRGB data
	  (this program can reading only RGBA png file)

	@XXX
	- if image size is small, segment fail occur

*/

void read_png(char *file_name)
{
	FILE *fp = fopen(file_name, "rb");

	if(!fp) abort();

	png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if(!png) abort();

	png_infop info = png_create_info_struct(png);
	if(!info) abort();

	if(setjmp(png_jmpbuf(png))) abort();

	png_init_io(png, fp);

	png_read_info(png, info);

	image_width 	= png_get_image_width(png, info);
	image_height	= png_get_image_height(png, info);
	color_type	= png_get_color_type(png, info);
	bit_depth	= png_get_bit_depth(png, info);

	if(bit_depth == 16)
		png_set_strip_16(png);

	if(color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_palette_to_rgb(png);

	if(color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8)
		png_set_expand_gray_1_2_4_to_8(png);

	if(png_get_valid(png, info, PNG_INFO_tRNS))
		png_set_tRNS_to_alpha(png);

	if(color_type == PNG_COLOR_TYPE_RGB ||
		color_type == PNG_COLOR_TYPE_GRAY ||
		color_type == PNG_COLOR_TYPE_PALETTE)
		png_set_filler(png, 0xff, PNG_FILLER_AFTER);
	
	row_pointers = (png_bytep*)malloc(sizeof(png_bytep) * image_height);

	int i;
	for(i = 0; i < image_height; i++)
		row_pointers[i] = (png_byte*)malloc(png_get_rowbytes(png, info));

	png_read_image(png, row_pointers);

	png_destroy_read_struct(&png, &info, NULL);
	png = NULL;
	info = NULL;

	fclose(fp);
}

void write_png(char *filename) {
	int i;

	FILE *fp = fopen(filename, "wb");
	if(!fp) abort();

	png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if(!png) abort();

	png_infop info = png_create_info_struct(png);
	if(!info) abort();

	if(setjmp(png_jmpbuf(png))) abort();

	png_init_io(png, fp);

	png_set_IHDR(
		png,
		info,
		image_width, image_height,
		8,
		PNG_COLOR_TYPE_RGBA,
		PNG_INTERLACE_NONE,
		PNG_COMPRESSION_TYPE_DEFAULT,
		PNG_FILTER_TYPE_DEFAULT
	);
	
	png_write_info(png, info);

	png_write_image(png, row_pointers);
	png_write_end(png, NULL);

	for(i = 0; i < image_height; i++)
		free(row_pointers[i]);

	free(row_pointers);

	if (png && info)
		png_destroy_write_struct(&png, &info);

	fclose(fp);
}

void pixel_data_extraction() 
{
	int x,y;
	int r, g, b ,a=0;
	int channel = 4;

	for(y = 0; y < image_height && color_type >= 4; y++)
	{

		for(x = 0; x < image_width; x++)
		{
			r = row_pointers[y][x*channel + 0];
			g = row_pointers[y][x*channel + 1];
			b = row_pointers[y][x*channel + 2];
			a = row_pointers[y][x*channel + 3];
	
			printf("(%d, %d) = [%d, %d, %d, %d]\n", x, y, r, g, b, a); 
		}
	}
}

int main(int argc, char *argv[]) 
{
	char *input_file = "test.png";
	char *output_file = "output.png";

	if(argc == 2)
	{
		input_file = argv[1];
	}
	if(argc == 3)
	{
		input_file = argv[1];
		output_file = argv[2];
	}

	read_png(input_file);
	pixel_data_extraction();
	write_png(output_file);

	printf("Program Terminated!!\n");

	return 0;
}
