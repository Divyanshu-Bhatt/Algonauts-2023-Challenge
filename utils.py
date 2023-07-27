import os
import sys
import logging


def get_logger(level=logging.INFO):
    log = logging.getLogger(__name__)
    if log.handlers:
        return log
    log.setLevel(level)
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s %(message)s", datefmt="%m/%d/%Y %I:%M:%S"
    )
    ch.setFormatter(formatter)
    log.addHandler(ch)
    return log


class argObj(object):
    def __init__(self, data_dir, parent_submission_dir, subj):
        self.subj = format(subj, "02")  # choosing sujbect

        self.data_dir = os.path.join(data_dir, "subj" + self.subj)
        self.parent_submission_dir = parent_submission_dir
        self.subject_submission_dir = os.path.join(
            self.parent_submission_dir, "subj" + self.subj
        )
        self.utils_dir = os.path.join(os.getcwd(), "utils", "subj" + self.subj)

        # contains the images (input)
        self.train_img_dir = os.path.join(
            self.data_dir, "training_split", "training_images"
        )
        self.test_img_dir = os.path.join(self.data_dir, "test_split", "test_images")

        # contains the output coordinates
        self.fmri_dir = os.path.join(self.data_dir, "training_split", "training_fmri")

        if not os.path.isdir(self.parent_submission_dir):
            os.makedirs(self.parent_submission_dir)

        if not os.path.isdir(self.subject_submission_dir):
            os.makedirs(self.subject_submission_dir)

        if not os.path.isdir(self.utils_dir):
            os.makedirs(self.utils_dir)
