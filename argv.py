from argparse import ArgumentParser

class pipe_parser(ArgumentParser):
    def __init__(self, parser):
        super().__init__(ArgumentParser)
        self.parser = parser
        self.parser.add_argument("-i", "--input", type=str, nargs=1, required=True,
                    help="[Necessary]Input filefold. The absolute path to your filefold which contains your raw images or relative path if it is in the same filefold as this python script. It must end with a '/'(in linux).")

        self.parser.add_argument("-o", "--output", type=str, nargs=1, required=True,
                               help="[Necessary]Output filefold. The absolute path to filefold which saves processed images or relative path if it is in the same filefold as this python script. If the filefold does not exist, it will create one. It must end with a '/'(in linux).")

        self.parser.add_argument("-m", "--mpp", type=float, nargs=1, required=True, default=0.1,
                           help="[Necessary]mpp: microns per pixon, the physical distance of a pixon. Default:0.1(for MERFISH images).It is key parameter of Deepcell model. If it performs bad on your cell images, it is very likely that the value of mpp is inappropriate. So we recommend that you can try some value to test on your images.")

        self.parser.add_argument("-z", "--zdistance", type=float, nargs=1, required=False, 
                           help="[Optional]It is the physical distance between two z-stacks, and the unit is micron. There is no need adding this parameter if your images are formed in chronological order.")

        self.parser.add_argument("-c", "--csv", type=str, nargs=1, required=False,
                           help="[Necessary]Output csv file. The absolute path to save the .csv file which contains property of cells in your images or relative path if it is in the same filefold as this python script. ")

        self.parser.add_argument("-f", "--format", type=str, nargs='+', required=False,
                           help="[Optional]Default:None. You can add one or more image format name after '-f', such as '-f png tif' and '-f tiff'. This will tell Deepcell to predict the images with the format you specified only in your input filefold path. If this paramater is not given, then Deepcell will try to predict all your images in your filefold. ")

        self.parser.add_argument("-g", "--gif", required=False, type=int, nargs=1,
                            help="[Optional]Add this parameter, DeepCell will create a gif file of spatially or chronologically sorted masks, which will be saved in your output filefold. You need to use '-g 6' to set fps to 6. Input bigger number to make gif play faster.")


        self.parser.add_argument("-k", "--ksize", required=False, type=tuple, nargs=1,
        help="[Optional]Kernal size of eroding and diluting. Add this parameter, along with erode times and dilute times, celltracker will  perform opening operation on images to track better.")

        self.parser.add_argument("-e", "--ero", required=False, type=int, nargs=1,
                        help="[Optional]Iteration times of eroding. Add this parameter, along with erode times and dilute times, celltracker will  perform opening operation on images to track better.")

        self.parser.add_argument("-d", "--dil", required=False, type=int, nargs=1,
                           help="[Optional]Iteration times of diluting. Add this parameter, along with erode times and dilute times, celltracker will  perform opening operation on images to track better.")

        self.parser.add_argument("-a", "--method", type=str, nargs='+', required=False, default='otsu',
                            help="[Optional]Default:'otsu'. Threshold picking strategy to decide the threshold in coordinates distance matrix. The other option:'var'.")
        
    def args(self):
        return self.parser.parse_args()