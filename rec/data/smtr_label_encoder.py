import numpy as np
import tensorflow as tf
import re
import copy
import random
import os

def add_special_char(dict_character, smtr_mode=False):
    if smtr_mode:
        EOS = '</s>'
        BOS = '<s>'
        IN_F = '<INF>'
        IN_B = '<INB>'
        PAD = '<pad>'
        dict_character = [EOS] + dict_character + [BOS, IN_F, IN_B, PAD]
    return dict_character

def create_dict_from_characters(character_list):
    """Create a dictionary mapping characters to indices."""
    char_dict = {}
    for i, char in enumerate(character_list):
        char_dict[char] = i
    return char_dict

def encode_base(text, char_dict, max_text_len, lower=False):
    """Basic encoding function converting text to indices."""
    if len(text) == 0:
        return None
    if lower:
        text = text.lower()

    text_list = []
    for char in text:
        if char not in char_dict:
            continue
        text_list.append(char_dict[char])

    if len(text_list) == 0 or len(text_list) > max_text_len:
        print(f'Invalid text length: {len(text)}, text is: {text}')
        return None

    return text_list

class SMTRLabelEncoder:
    def __init__(self, dict_character, use_space_char=True, max_text_length=25, sub_str_len=5):      
        # Store parameters
        self.max_text_length = max_text_length
        self.sub_str_len = sub_str_len

        # Define special tokens
        self.BOS = '<s>'
        self.EOS = '</s>'
        self.IN_F = '<INF>'
        self.IN_B = '<INB>'
        self.PAD = '<pad>'

        # Create character dictionary with special tokens
        if use_space_char and " " not in self.dict_character:
            self.dict_character.append(" ")
        self.dict_character = add_special_char(dict_character, smtr_mode=True)        

        # Create char to index mapping
        self.char_dict = create_dict_from_characters(self.dict_character)
        print(f'char to index: {self.char_dict}')
        self.num_character = len(self.char_dict)

    def encode(self, text, lower=False):
        """
        Encode text using SMTR encoding method

        Parameters:
        -----------
        text: str
            Text to encode
        lower: bool
            Whether to convert text to lowercase

        Returns:
        --------
        tuple or None
            Tuple containing encoded data, or None if text is invalid
        """
        # Initialize result dictionary
        data = {'label': text}

        # Encode text
        encoded_text = encode_base(text, self.char_dict, self.max_text_length, lower)
        if encoded_text is None:
            print(f'encoded_text is None in smtr label encode, text is: <{text}>')
            return None
        if len(encoded_text) > self.max_text_length:
            print(f'len(encoded_text) > self.max_text_length in smtr label encode, text is: {text}')
            return None

        # Set data values
        data['length'] = np.array(len(encoded_text))

        # Create input text with special tokens
        text_in = [self.char_dict[self.IN_F]] * self.sub_str_len + encoded_text + [self.char_dict[self.IN_B]] * self.sub_str_len

        # Initialize lists for substrings and next labels
        sub_string_list_pre = []  # 5 char before follow right to left
        next_label_pre = []       # 1 char after follow right to left
        sub_string_list = []      # 5 char before follow left to right
        next_label = []           # 1 char after follow left to right

        # Generate substrings and next labels
        for i in range(self.sub_str_len, len(text_in) - self.sub_str_len):
            sub_string_list.append(text_in[i - self.sub_str_len:i])
            next_label.append(text_in[i])

            if self.sub_str_len - i == 0:
                sub_string_list_pre.append(text_in[-i:])
            else:
                sub_string_list_pre.append(text_in[-i:self.sub_str_len - i])

            next_label_pre.append(text_in[-(i + 1)])

        # Add final substrings and labels
        sub_string_list.append(
            [self.char_dict[self.IN_F]] * (self.sub_str_len - len(encoded_text[-self.sub_str_len:])) +
            encoded_text[-self.sub_str_len:]
        )
        next_label.append(self.char_dict[self.EOS])

        sub_string_list_pre.append(
            encoded_text[:self.sub_str_len] +
            [self.char_dict[self.IN_B]] * (self.sub_str_len - len(encoded_text[:self.sub_str_len]))
        )
        next_label_pre.append(self.char_dict[self.EOS])

        # Generate range for random substitutions
        rang_subs = [i for i in range(1, self.sub_str_len + 1)]

        # Create data augmentation with random substitutions
        for sstr, l in zip(sub_string_list[self.sub_str_len:], next_label[self.sub_str_len:]):
            id_shu = np.random.choice(rang_subs, 2)

            sstr1 = copy.deepcopy(sstr)
            sstr1[id_shu[0] - 1] = random.randint(1, self.num_character - 5)
            if sstr1 not in sub_string_list:
                sub_string_list.append(sstr1)
                next_label.append(l)

            sstr[id_shu[1] - 1] = random.randint(1, self.num_character - 5)

        # Create more data augmentation
        for sstr, l in zip(sub_string_list_pre[self.sub_str_len:], next_label_pre[self.sub_str_len:]):
            id_shu = np.random.choice(rang_subs, 2)

            sstr1 = copy.deepcopy(sstr)
            sstr1[id_shu[0] - 1] = random.randint(1, self.num_character - 5)
            if sstr1 not in sub_string_list_pre:
                sub_string_list_pre.append(sstr1)
                next_label_pre.append(l)

            sstr[id_shu[1] - 1] = random.randint(1, self.num_character - 5)

        # Store lengths and pad sequences
        data['length_subs'] = np.array(len(sub_string_list))

        pad_length = (self.max_text_length * 2) + 2 - len(sub_string_list)
        sub_string_list_padded = sub_string_list + [[self.char_dict[self.PAD]] * self.sub_str_len] * pad_length
        next_label_padded = next_label + [self.char_dict[self.PAD]] * ((self.max_text_length * 2) + 2 - len(next_label))

        data['label_subs'] = np.array(sub_string_list_padded)
        data['label_next'] = np.array(next_label_padded)

        # Store lengths and pad sequences for pre-processing
        data['length_subs_pre'] = np.array(len(sub_string_list_pre))

        pad_length_pre = (self.max_text_length * 2) + 2 - len(sub_string_list_pre)
        sub_string_list_pre_padded = sub_string_list_pre + [[self.char_dict[self.PAD]] * self.sub_str_len] * pad_length_pre
        next_label_pre_padded = next_label_pre + [self.char_dict[self.PAD]] * ((self.max_text_length * 2) + 2 - len(next_label_pre))

        data['label_subs_pre'] = np.array(sub_string_list_pre_padded)
        data['label_next_pre'] = np.array(next_label_pre_padded)

        # Create final label with BOS and EOS tokens
        final_text = [self.char_dict[self.BOS]] + encoded_text + [self.char_dict[self.EOS]]
        final_text_padded = final_text + [self.char_dict[self.PAD]] * (self.max_text_length + 2 - len(final_text))
        data['label'] = np.array(final_text_padded)

        return (
            data['label'],
            data['label_subs'],
            data['label_next'],
            data['length_subs'],
            data['label_subs_pre'],
            data['label_next_pre'],
            data['length_subs_pre'],
            data['length']
        )