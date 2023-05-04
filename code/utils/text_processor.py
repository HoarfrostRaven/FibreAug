import json
import os
import augly.text as textaugs


class TextProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        # Create output directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        # Create txt file to store processing information
        with open(os.path.join(output_path, 'processing_info.json'), 'w') as f:
            f.write(self._process_info())

    def process(self, text_file):
        # Read in text file
        with open(os.path.join(self.input_path, text_file), "r", encoding="utf-8") as f:
            data = json.load(f)

        # Processing of captions
        processed_captions = []
        for item in data:
            caption = item["caption"]
            processed_caption = self._process(caption)
            item["caption"] = processed_caption
            processed_captions.append(item)

        # Save processed captions to file
        output_file_path = os.path.join(self.output_path, text_file)
        with open(output_file_path, "w", encoding="utf-8") as f:
            json.dump(processed_captions,
                      f,
                      indent=4,
                      separators=(',', ': '),
                      ensure_ascii=False
                      )
            f.write('\n')

    def _process(self, text):
        # Add text processing methods here
        # Return the processed text
        return text.upper()  # Example: Convert all text to uppercase

    def _process_info(self):
        # Return processing information
        return 'Default processing information'


class ChangeCase(TextProcessor):
    def __init__(self,
                 input_path,
                 output_path,
                 granularity: str = "all",
                 cadence: float = 1.0,
                 case: str = "random",
                 p: float = 1.0
                 ):
        """Change case for the text

        Args:
            input_path (str): Path of input file.
            output_path (str): Path of output file.
            granularity (str, optional): 
                    "char"(case of random chars is changed), 
                    "word"(case of random words is changed), 
                    "all"(case of the entire text is changed). 
                    Defaults to "all".
            cadence (float, optional): 
                    How frequent (i.e. between this many characters/words) to change the case. 
                    Must be at least 1.0.
                    Non-integer values are used as an 'average' cadence. 
                    Not used for granularity 'all'.
                    Defaults to 1.0.
            case (str, optional): 
                    The case to change words to; 
                    valid values are 'lower', 'upper', 'title', or 'random' 
                    Defaults to "random".
            p (float, optional): 
                    The probability of the transform being applied.
                    Defaults to 1.0.
        """
        self.granularity = granularity
        self.cadence = cadence
        self.case = case
        self.p = p
        super().__init__(input_path, os.path.join(output_path, "ChangeCase"))

        self.aug = textaugs.ChangeCase(granularity=self.granularity,
                                       cadence=self.cadence,
                                       case=self.case,
                                       p=self.p
                                       )

    def _process(self, text):
        # Change case for the text
        return self.aug(text)

    def _process_info(self):
        info_dict = {"ChangeCase": []}

        # Add information for each parameter
        info = [{"key": "input_path", "value": self.input_path},
                {"key": "output_path", "value": self.output_path},
                {"key": "granularity", "value": str(self.granularity)},
                {"key": "cadence", "value": str(self.cadence)},
                {"key": "case", "value": str(self.case)},
                {"key": "p", "value": str(self.p)}]
        info_dict["ChangeCase"].extend(info)

        # Convert dictionary to JSON string and return
        return json.dumps(info_dict, indent=4)
