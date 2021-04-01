"""
Sagitta - Young star classification and age regression pipeline
Paper: Untangling the Galaxy III: Photometric Search for Pre-main Sequence Stars with Deep Learning
Authors: Aidan McBride, Ryan Lingg, Marina Kounkel, Kevin Covey, and Brian Hutchinson
"""

#--------------- External Imports ---------------#
import os
import sys
import argparse
import galpy.util.bovy_coords as bc
import numpy as np
import pandas as pd
import torch
import torch.utils.data
from astropy.table import Table as AstroTable

#--------------- Local Imports ---------------#
if __name__ == "sagitta.sagitta":
    from sagitta.data_tools import DataTools, SagittaDataset
    from sagitta.model_code import Sagitta
else:
    from data_tools import DataTools, SagittaDataset
    from model_code import Sagitta

#--------------- Main Pipeline ---------------#
def main():
    """
    Runs the pipeline
    """
    pipeline = SagittaPipeline(args=parse_args())
    if not pipeline.check_valid_args_config():
        sys.exit()
    pipeline.get_naming_changes()
    pipeline.load_input_table()
    #if not pipeline.args.ignore_bad_rows:
    #    pipeline.check_bad_source_id_rows()
    if pipeline.check_output_columns_overwrite():
        sys.exit()
    pipeline.try_ra_dec_to_l_b()
    pipeline.download_missing_data()
    pipeline.data_frame = DataTools.fix_and_mark_nans(
                                        data_frame=pipeline.data_frame,
                                        nan_column_extension=pipeline.nan_column_suffix
                                        )
    if not pipeline.args.download_only:
        if pipeline.should_run_av_model():
            pipeline.predict_av()
        if pipeline.should_run_pms_model():
            pipeline.predict_pms()
        if pipeline.should_run_age_model():
            pipeline.predict_age()
        if pipeline.should_generate_av_uncertainties():
            pipeline.generate_av_uncertainties()
        if pipeline.should_generate_pms_uncertainties():
            pipeline.generate_pms_uncertainties()
        if pipeline.should_generate_age_uncertainties():
            pipeline.generate_age_uncertainties()
    pipeline.save_output_table()
    print("Done")

#--------------- Sagitta Pipeline Class ---------------#
class SagittaPipeline:
    """
    The pipeline is run by calling the functions
    in this class and modifying its internal state
    """

    def __init__(self, args):
        """
        1) Parses the command line arguments
        2) Sets the directory location for the model state dicts
        3) Creates placeholders for long standing variables
        4) Sets the pytorch device that will be used for prediction
        """
        sagitta_path, _ = os.path.split(os.path.abspath(__file__))
        self.state_dict_dir = os.path.join(sagitta_path, "model_state_dicts")
        self.args = args
        self.nan_column_suffix = "_nan_mask"
        self.overlap_column_suffix = "_overlap"
        self.std_input_col_naming = {}
        self.working_overlap_column_renaming = {}
        self.data_frame = None
        self.model_output_frame = None

    def get_naming_changes(self):
        """
        Builds a dictionary for converting between standard
        column names and user given input table column names
        """
        self.std_input_col_naming = {
            "source_id" :   self.args.source_id.lower(),
            "l"         :   self.args.l.lower(),
            "b"         :   self.args.b.lower(),
            "ra"        :   self.args.ra.lower(),
            "dec"       :   self.args.dec.lower(),
            "g"         :   self.args.g.lower(),
            "bp"        :   self.args.bp.lower(),
            "rp"        :   self.args.rp.lower(),
            "j"         :   self.args.j.lower(),
            "h"         :   self.args.h.lower(),
            "k"         :   self.args.k.lower(),
            "parallax"  :   self.args.parallax.lower(),
            "eg"        :   self.args.eg.lower(),
            "ebp"       :   self.args.ebp.lower(),
            "erp"       :   self.args.erp.lower(),
            "ej"        :   self.args.ej.lower(),
            "eh"        :   self.args.eh.lower(),
            "ek"        :   self.args.ek.lower(),
            "eparallax" :   self.args.eparallax.lower()
        }
        if self.args.av:
            self.std_input_col_naming["av"] = self.args.av.lower()


    def load_input_table(self):
        """
        1) Loads the user given input table
        2) Checks that user given column names match
        3) Entering test mode
        4) Creating input pandas frame
        """
        print("Loading input table")
        if not self.args.single_object:
            input_table = AstroTable.read(self.args.tableIn)
            input_table_column_names = [col.lower() for col in input_table.colnames]
            user_specified_columns = []
            for _, (std, user_given) in enumerate(self.std_input_col_naming.items()):
                if std != user_given:
                    user_specified_columns.append(user_given)
            incorrect_names = set(user_specified_columns)-set(input_table_column_names)
            if self.args.av and self.args.av not in input_table_column_names:
                incorrect_names.add(self.args.av)
            if len(incorrect_names) > 0:
                print("ERROR: Unable to find the specified column(s) in the input table:",
                        ", ".join(list(incorrect_names)),
                        file=sys.stderr)
                sys.exit()
            if self.args.test:
                print("In testing mode!")
                input_table = input_table[:10000]
            print("\tMaking dataframe from table")
            self.data_frame = DataTools.pandas_from_table(table=input_table)
        else:
            print('Creating a catalog using a given source_id')
            self.data_frame = pd.DataFrame(data=[int(self.args.tableIn)],columns=['source_id'])


    def check_bad_source_id_rows(self):
        """
        Determines if there are any issues with the values of the source id column
        """
        source_id_col = self.std_input_col_naming["source_id"]
        valid_id_rows = np.where((np.logical_not(np.isnan(self.data_frame[source_id_col]))) & (self.data_frame[source_id_col]>0))[0]
        if len(valid_id_rows) != len(self.data_frame):
            print("ERROR: The input table must have source IDs for every row.",
                    file=sys.stderr)
            sys.exit()
        if not self.data_frame[source_id_col].dropna().is_unique:
            print("WARNING: There are duplcicates for the source IDs in the input table.",
                    file=sys.stderr)
            answer = ""
            while (answer.lower() != "continue" and answer.lower() != "exit"):
                answer = input("To continue, confirm that you are willing to drop rows with " +
                                "duplicate Gaia source IDs? (continue/exit).\n").strip()
                if answer == "continue":
                    self.data_frame.drop_duplicates(subset=source_id_col, inplace=True)
                else:
                    print("Aborting. To ignore, run with --ignore_bad_rows")
                    sys.exit()


    def check_output_columns_overwrite(self):
        """
        Checks that no output columns will overwrite current columns
        """
        overwrite = False
        stats_suffixes = ["_mean", "_median", "_std", "_var", "_min", "_max"]
        av_stats_out_columns = [self.args.av_out+suffix for suffix in stats_suffixes]
        pms_stats_out_columns = [self.args.pms_out+suffix for suffix in stats_suffixes]
        age_stats_out_columns = [self.args.age_out+suffix for suffix in stats_suffixes]
        if self.should_run_av_model():
            if self.args.av_out in self.data_frame.columns:
                print("ERROR: The specified av output prediction column " +
                        "({}) is already in the table. Please use the ".format(self.args.av_out) +
                        "--av_out flag to choose a new output column name, specify this as " +
                        "the input av column with the --av flag, or use the --no_av_prediction " +
                        "flag to not run the av model.")
                overwrite = True
        if self.should_run_pms_model():
            if self.args.pms_out in self.data_frame.columns:
                print("ERROR: The specified pms output prediction column " +
                        "({}) is already in the table. Please use the ".format(self.args.pms_out) +
                        "--pms_out flag and specify a new unused column name, or " +
                        "use the --no_pms_prediction flag to not run the pms model." )
                overwrite = True
        if self.should_run_age_model():
            if self.args.age_out in self.data_frame.columns:
                print("ERROR: The specified age output prediction column " +
                        "({}) is already in the table. Please use the ".format(self.args.age_out) +
                        "--age_out flag and specify a new unused column name, or " +
                        "use the --no_age_prediction flag to not run the age model." )
                overwrite = True
        if self.should_generate_av_uncertainties():
            for column_name in av_stats_out_columns:
                if column_name in self.data_frame.columns:
                    print("ERROR: The specified av uncertainty output prediction column " +
                        "({}) is already in the table. Please use the ".format(column_name) +
                        "--av_out flag and specify a new unused column name.")
                    overwrite = True
        if self.should_generate_pms_uncertainties():
            for column_name in pms_stats_out_columns:
                if column_name in self.data_frame.columns:
                    print("ERROR: The specified pms uncertainty output prediction column " +
                        "({}) is already in the table. Please use the ".format(column_name) +
                        "--pms_out flag and specify a new unused column name.")
                    overwrite = True
            for column_name in age_stats_out_columns:
                if column_name in self.data_frame.columns:
                    print("ERROR: The specified age uncertainty output prediction column " +
                        "({}) is already in the table. Please use the ".format(column_name) +
                        "--age_out flag and specify a new unused column name.")
                    overwrite = True
        return overwrite


    def try_ra_dec_to_l_b(self):
        """
        1) Look to see if ra and dec are included but not l and b
        2) If so convert ra and dec to l and b
        3) Store l and b in the input table
        """
        ra_name = self.std_input_col_naming["ra"]
        dec_name = self.std_input_col_naming["dec"]
        l_name = self.std_input_col_naming["l"]
        b_name = self.std_input_col_naming["b"]
        if all(x in self.data_frame.columns for x in [ra_name, dec_name]):
            if not all(x in self.data_frame.columns for x in [l_name, b_name]):
                print("Converting RA & Dec to L & B")
                l_b = bc.radec_to_lb(
                            self.data_frame[ra_name],
                            self.data_frame[dec_name],
                            degree=True
                            )
                self.data_frame["l"] = l_b[:,0]
                self.data_frame["b"] = l_b[:,1]

    def download_missing_data(self):
        """
        1) Gets the missing required field names
        2) If fields are missing, uses Astroquery to download them
        3) Then appends them to the input dataframe
        """
        missing_fields = self.get_missing_fields_list()
        if len(missing_fields) > 0:
            missing_fields_frame = pd.DataFrame()
            source_id_col_name = self.std_input_col_naming["source_id"]
            missing_fields_frame["source_id"] = self.data_frame[source_id_col_name]
            missing_fields_frame = DataTools.download_missing_fields(
                                                    data_frame=missing_fields_frame,
                                                    missing_fields=missing_fields,
                                                    ver=self.args.version
                                                    )
            missing_fields_frame.rename(
                                    columns={"source_id" : self.std_input_col_naming["source_id"]},
                                    inplace=True
                                    )
            missing_fields_frame=missing_fields_frame.drop_duplicates()
            self.data_frame = self.data_frame.merge(
                                                missing_fields_frame,
                                                on=self.std_input_col_naming["source_id"],
                                                how="left"
                                                )

    def get_missing_fields_list(self):
        """
        1) Checks the columns in the frame against the list of required columns
        2) Returns any columns that were not found
        """
        frame_columns = list(self.data_frame.columns)
        for _, (working, given) in enumerate(self.std_input_col_naming.items()):
            if given in frame_columns:
                frame_columns.remove(given)
                frame_columns.append(working)
        important_columns = {"source_id"}
        if self.args.download_only:
            for field in ["parallax", "g", "bp", "rp", "j", "h", "k",
                            "eparallax", "eg", "ebp", "erp", "ej", "eh", "ek",
                            "pmra", "pmdec", "epmra", "epmdec"]:
                important_columns.add(field)
        else:
            if self.should_run_av_model():
                for field in ["l", "b", "parallax"]:
                    important_columns.add(field)
            if self.should_generate_av_uncertainties():
                for field in ["l", "b", "parallax", "eparallax"]:
                    important_columns.add(field)
            if self.should_run_pms_model() or self.should_run_age_model():
                for field in ["parallax", "g", "bp", "rp", "j", "h", "k"]:
                    important_columns.add(field)
            if self.should_generate_pms_uncertainties() or self.should_generate_age_uncertainties():
                for field in ["parallax", "g", "bp", "rp", "j", "h", "k",
                                "eparallax", "eg", "ebp", "erp", "ej", "eh", "ek"]:
                    important_columns.add(field)
        missing_fields = []
        for col in important_columns:
            if col not in frame_columns:
                missing_fields.append(col)
        return missing_fields

    def predict_av(self):
        """
        1) Runs the data through the Av model to generate values
        2) Appends the Av values to the dataframe
        """
        print("Predicting stellar extinctions as {}".format(self.args.av_out))
        av_dataset = SagittaDataset(
                                frame=self.data_frame,
                                data_format="StellarExtinction",
                                column_names=self.std_input_col_naming
                                )
        av_dataloader = torch.utils.data.DataLoader(
                                        av_dataset,
                                        batch_size=self.args.batch_size,
                                        shuffle=False,
                                        num_workers=os.cpu_count()//2,
                                        drop_last=False
                                        )
        av_model = Sagitta(connectShape=40)
        av_model.load_state_dict(
                            torch.load(
                                os.path.join(self.state_dict_dir, "av_model.pt"),
                                map_location=torch.device(self.args.device)
                                )
                            )
        av_model.to(torch.device(self.args.device))
        av_model.eval()
        av_model_output = []
        for idx, section_input in enumerate(av_dataloader):
            section_input = section_input.to(torch.device(self.args.device))
            with torch.no_grad():
                print("\t{:.1f}% completed".format(idx/len(av_dataloader)*100), end="\r")
                section_output = av_model(section_input)
                av_model_output.append(section_output)
        av_model_output = torch.cat(av_model_output).cpu().detach().numpy()*5
        self.data_frame[self.args.av_out] = av_model_output
        self.std_input_col_naming["av"] = self.args.av_out

    def generate_av_uncertainties(self):
        """
        1) Scatters the input fields in a random normal way based off of uncertainties
        2) Runs the data through the Av model to generate model uncertainty values
        3) Aggregates the outputs into summerized Av uncertainty values
        4) Appends the summerized Av unrecertainty values to the dataframe
            (drops any previous stat columns with the same names)
        """
        print("Generating Av uncertainties with " + str(self.args.av_uncertainty) +
                " samples per star\n\tScattering Av model inputs")
        loop_frame = pd.DataFrame()
        varied_frame = pd.DataFrame()
        l_name = self.std_input_col_naming["l"]
        b_name = self.std_input_col_naming["b"]
        source_id_name = self.std_input_col_naming["source_id"]
        parallax_name = self.std_input_col_naming["parallax"]
        eparallax_name = self.std_input_col_naming["eparallax"]
        std_naming = {
            "l"         :   "l",
            "b"         :   "b",
            "source_id" :   "source_id",
            "parallax"  :   "parallax",
            "eparallax" :   "eparallax",
        }
        for idx in range(self.args.av_uncertainty):
            print("\t{:.1f}% completed".format(idx/self.args.av_uncertainty*100), end="\r")
            loop_frame["parallax"] = self.data_frame[parallax_name].to_numpy() + (
                                        np.random.normal(size=len(self.data_frame))*
                                        self.data_frame[eparallax_name].to_numpy()
                                        )
            loop_frame["l"] = self.data_frame[l_name]
            loop_frame["b"] = self.data_frame[b_name]
            loop_frame["source_id"] = self.data_frame[source_id_name]
            varied_frame = varied_frame.append(loop_frame.copy(), ignore_index=True)
        print("\tRunning the Av model on the varied data")
        av_uncertainty_dataset = SagittaDataset(
                                        frame=varied_frame,
                                        data_format="StellarExtinction",
                                        column_names=std_naming,
                                        )
        av_uncertainty_dataloader = torch.utils.data.DataLoader(
                                                av_uncertainty_dataset,
                                                batch_size=self.args.batch_size,
                                                shuffle=False,
                                                num_workers=os.cpu_count()//2,
                                                drop_last=False
                                                )
        av_model = Sagitta(connectShape=40)
        av_model.load_state_dict(
                            torch.load(
                                os.path.join(self.state_dict_dir, "av_model.pt"),
                                map_location=torch.device(self.args.device)
                                )
                            )
        av_model.to(torch.device(self.args.device))
        av_model.eval()
        av_predictions = []
        for idx, section_input in enumerate(av_uncertainty_dataloader):
            section_input = section_input.to(torch.device(self.args.device))
            with torch.no_grad():
                print("\t{:.1f}% completed".format(idx/len(av_uncertainty_dataloader)*100),
                        end="\r")
                section_output = av_model(section_input)
                av_predictions.append(section_output)
        varied_frame["av"] = torch.cat(av_predictions).cpu().detach().numpy()*5
        print("\tAggregating the Av uncertainties")
        av_stats_frame = varied_frame.filter(items=["source_id", "av"]).groupby("source_id")
        av_stats_frame = av_stats_frame.agg(["mean", "median", "std", "var", "min", "max"])
        av_stats_frame = av_stats_frame["av"].reset_index()
        av_stats_frame = av_stats_frame.filter(items=["source_id", "mean", "median",
                                                        "std", "var", "min", "max"])
        av_stats_frame.rename(inplace=True,
                                columns={
                                    "mean"      :   self.args.av_out + "_mean",
                                    "median"    :   self.args.av_out + "_median",
                                    "std"       :   self.args.av_out + "_std",
                                    "var"       :   self.args.av_out + "_var",
                                    "min"       :   self.args.av_out + "_min",
                                    "max"       :   self.args.av_out + "_max",
                                    "source_id" :   self.std_input_col_naming["source_id"]
                                })
        self.data_frame = self.data_frame.merge(
                                        av_stats_frame,
                                        how="inner",
                                        on=self.std_input_col_naming["source_id"]
                                        )

    def predict_pms(self):
        """
        1) Runs the data through the Yso model to generate values
        2) Appends the Yso values to the dataframe
        """
        print("Predicting PMS probablities as {}".format(self.args.pms_out))
        pms_dataset = SagittaDataset(
                                frame=self.data_frame,
                                data_format="PMSClassifier",
                                column_names=self.std_input_col_naming
                                )
        pms_dataloader = torch.utils.data.DataLoader(
                                            pms_dataset,
                                            batch_size=self.args.batch_size,
                                            shuffle=False,
                                            num_workers=os.cpu_count()//2,
                                            drop_last=False
                                            )
        pms_model = Sagitta()
        pms_model.load_state_dict(
                            torch.load(
                                os.path.join(self.state_dict_dir, "pms_model.pt"),
                                map_location=torch.device(self.args.device)
                                )
                            )
        pms_model.to(torch.device(self.args.device))
        pms_model.eval()
        pms_predictions = []
        for idx, section_input in enumerate(pms_dataloader):
            section_input = section_input.to(torch.device(self.args.device))
            with torch.no_grad():
                print("\t{:.1f}% completed".format(idx/len(pms_dataloader)*100), end="\r")
                section_output = torch.sigmoid(pms_model(section_input))
                pms_predictions.append(section_output)
        pms_predictions = torch.cat(pms_predictions).cpu().detach().numpy()
        self.data_frame[self.args.pms_out] = pms_predictions

    def generate_pms_uncertainties(self):
        """
        1) Scatters the input fields in a random normal way based off of uncertainties
        2) Runs the data through the Yso model to generate model uncertainty values
        3) Aggregates the outputs into summerized Yso uncertainty values
        4) Appends the summerized Yso uncertainty values to the dataframe
            (drops any previous stat columns with the same names)
        """
        print("Generating PMS uncertainties with " + str(self.args.pms_uncertainty) +
                " samples per star\n\tScattering PMS model inputs")
        loop_frame = pd.DataFrame()
        varied_frame = pd.DataFrame()
        input_std_col_naming = {v: k for k, v in self.std_input_col_naming.items()}
        for idx in range(self.args.pms_uncertainty):
            print("\t{:.1f}% completed".format(idx/self.args.pms_uncertainty*100), end="\r")
            val_err_list = [
                [self.std_input_col_naming["g"], self.std_input_col_naming["eg"]],
                [self.std_input_col_naming["bp"], self.std_input_col_naming["ebp"]],
                [self.std_input_col_naming["rp"], self.std_input_col_naming["erp"]],
                [self.std_input_col_naming["j"], self.std_input_col_naming["ej"]],
                [self.std_input_col_naming["h"], self.std_input_col_naming["eh"]],
                [self.std_input_col_naming["k"], self.std_input_col_naming["ek"]],
                [self.std_input_col_naming["parallax"], self.std_input_col_naming["eparallax"]]
            ]
            for val, err in val_err_list:
                loop_frame[input_std_col_naming[val]] = self.data_frame[val].to_numpy() + (
                                        np.random.normal(size=len(self.data_frame)) *
                                        self.data_frame[err].to_numpy()
                                        )
                loop_frame["av"] = self.data_frame[self.std_input_col_naming["av"]].to_numpy() + (
                                                            np.random.uniform(
                                                                low=-self.args.av_scatter_range,
                                                                high=self.args.av_scatter_range,
                                                                size=len(self.data_frame)
                                                            )
                                                        )
                loop_frame["source_id"] = self.data_frame[self.std_input_col_naming["source_id"]]
            varied_frame = varied_frame.append(loop_frame.copy(), ignore_index=True)
        print("\tRunning the PMS model on the varied data")
        std_naming = {
            "parallax"  :   "parallax",
            "av"        :   "av",
            "g"         :   "g",
            "bp"        :   "bp",
            "rp"        :   "rp",
            "j"         :   "j",
            "h"         :   "h",
            "k"         :   "k"
        }
        pms_uncertainty_dataset = SagittaDataset(
                                        frame=varied_frame,
                                        data_format="PMSClassifier",
                                        column_names=std_naming
                                        )
        pms_uncertainty_dataloader = torch.utils.data.DataLoader(
                                                pms_uncertainty_dataset,
                                                batch_size=self.args.batch_size,
                                                shuffle=False,
                                                num_workers=os.cpu_count()//2,
                                                drop_last=False
                                                )
        pms_model = Sagitta()
        pms_model.load_state_dict(
                            torch.load(
                                os.path.join(self.state_dict_dir, "pms_model.pt"),
                                map_location=torch.device(self.args.device)
                                )
                            )
        pms_model.to(torch.device(self.args.device))
        pms_model.eval()
        pms_predictions = []
        for idx, section_input in enumerate(pms_uncertainty_dataloader):
            section_input = section_input.to(torch.device(self.args.device))
            with torch.no_grad():
                print("\t{:.1f}% completed".format(idx/len(pms_uncertainty_dataloader)*100),
                        end="\r")
                section_output = torch.sigmoid(pms_model(section_input))
                pms_predictions.append(section_output)
        varied_frame["pms"] = torch.cat(pms_predictions).cpu().detach().numpy()
        print("\tAggregating the pms probablity uncertainties")
        pms_stats_frame = varied_frame.filter(items=["source_id", "pms"]).groupby("source_id")
        pms_stats_frame = pms_stats_frame.agg(["mean", "median", "std", "var", "min", "max"])
        pms_stats_frame = pms_stats_frame["pms"].reset_index()
        pms_stats_frame = pms_stats_frame.filter(items=["source_id", "mean", "median",
                                                "std", "var", "min", "max"])
        pms_stats_frame.rename(inplace=True,
                                columns={
                                    "mean"      :   self.args.pms_out + "_mean",
                                    "median"    :   self.args.pms_out + "_median",
                                    "std"       :   self.args.pms_out + "_std",
                                    "var"       :   self.args.pms_out + "_var",
                                    "min"       :   self.args.pms_out + "_min",
                                    "max"       :   self.args.pms_out + "_max",
                                    "source_id" :   self.std_input_col_naming["source_id"]
                                })
        self.data_frame = self.data_frame.merge(
                                            pms_stats_frame,
                                            how="inner",
                                            on=self.std_input_col_naming["source_id"]
                                            )

    def predict_age(self):
        """
        1) Runs the data through the Age model to generate values
        2) Appends the Age values to the dataframe
        """
        print("Predicting stellar ages as {}".format(self.args.age_out))
        age_dataset = SagittaDataset(
                                frame=self.data_frame,
                                data_format="YoungStarAgeRegressor",
                                column_names=self.std_input_col_naming
                                )
        age_dataloader = torch.utils.data.DataLoader(
                                                age_dataset,
                                                batch_size=self.args.batch_size,
                                                shuffle=False,
                                                num_workers=os.cpu_count()//2,
                                                drop_last=False
                                                )
        age_model = Sagitta()
        age_model.load_state_dict(
                            torch.load(
                                os.path.join(self.state_dict_dir, "age_model.pt"),
                                map_location=torch.device(self.args.device)
                                )
                            )
        age_model.to(torch.device(self.args.device))
        age_model.eval()
        age_predictions = []
        for idx, section_input in enumerate(age_dataloader):
            section_input = section_input.to(torch.device(self.args.device))
            with torch.no_grad():
                print("\t{:.1f}% completed".format(idx/len(age_dataloader)*100), end="\r")
                section_output = age_model(section_input)
                age_predictions.append(section_output)
        age_predictions = torch.cat(age_predictions).cpu().detach().numpy()
        age_predictions = DataTools.normalize_gaia(
                                        column_vals=age_predictions,
                                        column_name="age",
                                        back=True
                                        )
        self.data_frame[self.args.age_out] = age_predictions

    def generate_age_uncertainties(self):
        """
        1) Scatters the input fields in a random normal way based off of uncertainties
        2) Runs the data through the Age model to generate model uncertainty values
        3) Aggregates the outputs into summerized Age uncertainty values
        3) Appends the summerized Age uncertainty values to the dataframe
            (drops any previous stat columns with the same names)
        """
        print("Generating Age uncertainties with " + str(self.args.age_uncertainty) +
                " samples per star\n\tScattering Age model inputs")
        loop_frame = pd.DataFrame()
        varied_frame = pd.DataFrame()
        input_std_col_naming = {v: k for k, v in self.std_input_col_naming.items()}
        for idx in range(self.args.age_uncertainty):
            print("\t{:.1f}% completed".format(idx/self.args.age_uncertainty*100), end="\r")
            val_err_list = [
                [self.std_input_col_naming["g"], self.std_input_col_naming["eg"]],
                [self.std_input_col_naming["bp"], self.std_input_col_naming["ebp"]],
                [self.std_input_col_naming["rp"], self.std_input_col_naming["erp"]],
                [self.std_input_col_naming["j"], self.std_input_col_naming["ej"]],
                [self.std_input_col_naming["h"], self.std_input_col_naming["eh"]],
                [self.std_input_col_naming["k"], self.std_input_col_naming["ek"]],
                [self.std_input_col_naming["parallax"], self.std_input_col_naming["eparallax"]]
            ]
            for val, err in val_err_list:
                loop_frame[input_std_col_naming[val]] = self.data_frame[val].to_numpy() + (
                                        np.random.normal(size=len(self.data_frame)) *
                                        self.data_frame[err].to_numpy()
                                        )
                loop_frame["av"] = self.data_frame[self.std_input_col_naming["av"]].to_numpy() + (
                                                            np.random.uniform(
                                                                low=-self.args.av_scatter_range,
                                                                high=self.args.av_scatter_range,
                                                                size=len(self.data_frame)
                                                            )
                                                        )
                loop_frame["source_id"] = self.data_frame[self.std_input_col_naming["source_id"]]
            varied_frame = varied_frame.append(loop_frame.copy(), ignore_index=True)
        std_naming = {
            "parallax"  :   "parallax",
            "av"        :   "av",
            "g"         :   "g",
            "bp"        :   "bp",
            "rp"        :   "rp",
            "j"         :   "j",
            "h"         :   "h",
            "k"         :   "k"
        }
        print("\tRunning the age model on the varied data")
        age_uncertainty_dataset = SagittaDataset(
                                        frame=varied_frame,
                                        data_format="YoungStarAgeRegressor",
                                        column_names=std_naming
                                        )
        age_uncertainty_dataloader = torch.utils.data.DataLoader(
                                                age_uncertainty_dataset,
                                                batch_size=self.args.batch_size,
                                                shuffle=False,
                                                num_workers=os.cpu_count()//2,
                                                drop_last=False
                                                )
        age_model = Sagitta()
        age_model.load_state_dict(
                            torch.load(
                                os.path.join(self.state_dict_dir, "age_model.pt"),
                                map_location=torch.device(self.args.device)
                                )
                            )
        age_model.to(torch.device(self.args.device))
        age_model.eval()
        age_predictions = []
        for idx, section_input in enumerate(age_uncertainty_dataloader):
            section_input = section_input.to(torch.device(self.args.device))
            with torch.no_grad():
                print("\t{:.1f}% completed".format(idx/len(age_uncertainty_dataloader)*100),
                        end="\r")
                section_output = age_model(section_input)
                age_predictions.append(section_output)
        age_predictions = torch.cat(age_predictions).cpu().detach().numpy()
        varied_frame["age"] = DataTools.normalize_gaia(
                                            column_vals=age_predictions,
                                            column_name="age",
                                            back=True
                                            )
        print("\tAggregating the age uncertainties")
        age_stats_frame = varied_frame.filter(items=["source_id", "age"]).groupby("source_id")
        age_stats_frame = age_stats_frame.agg(["mean", "median", "std", "var", "min", "max"])
        age_stats_frame = age_stats_frame["age"].reset_index()
        age_stats_frame = age_stats_frame.filter(items=["source_id", "mean", "median",
                                                "std", "var", "min", "max"])
        age_stats_frame.rename(inplace=True,
                                columns={
                                    "mean"      :   self.args.age_out + "_mean",
                                    "median"    :   self.args.age_out + "_median",
                                    "std"       :   self.args.age_out + "_std",
                                    "var"       :   self.args.age_out + "_var",
                                    "min"       :   self.args.age_out + "_min",
                                    "max"       :   self.args.age_out + "_max",
                                    "source_id" :   self.std_input_col_naming["source_id"]
                                })
        self.data_frame = self.data_frame.merge(
                                            age_stats_frame,
                                            how="inner",
                                            on=self.std_input_col_naming["source_id"]
                                            )

    def save_output_table(self):
        """
        1) Created the final astropy output masked table object
        2) Inserts the masks over the columns that have nan masks
        3) Removes the nan mask columns
        4) Sets the output table file name
        5) Performs the writing of the table to disk
        """
        output_table = AstroTable(AstroTable.from_pandas(self.data_frame), masked=True)
        for column_name in output_table.colnames:
            column_nan_mask_name = column_name + self.nan_column_suffix
            if column_nan_mask_name in output_table.colnames:
                output_table[column_name] = AstroTable.MaskedColumn(
                                                        data=output_table[column_name],
                                                        name=column_name,
                                                        mask=output_table[column_nan_mask_name]
                                                        )
                output_table.remove_column(column_nan_mask_name)
        output_table_name = self.get_output_table_name()
        if self.args.test:
            print("Output table columns:", ", ".join(output_table.colnames))
        print("Saving output table to {}".format(output_table_name))
        
        a=np.where(output_table[self.std_input_col_naming["source_id"]]<0)[0]
        output_table[self.std_input_col_naming["source_id"]].mask[a]=True
        output_table.write(output_table_name, overwrite=True)

    def get_output_table_name(self):
        """
        Returns the output fits file name
        """
        if self.args.tableOut:
            return self.args.tableOut
        _, input_filename = os.path.split(self.args.tableIn)
        input_filename_prefix = input_filename.split(".")[0]
        if self.args.test:
            input_filename_prefix += "-test"
        return input_filename_prefix + "-sagitta.fits"

    def should_run_av_model(self):
        """
        Flow control for running the Av rergression model
        """
        if self.args.no_av_prediction or self.args.av:
            return False
        return True

    def should_generate_av_uncertainties(self):
        """
        Flow control for generating Av rergression model uncertainties
        """
        if self.args.av_uncertainty > 0:
            return True
        return False

    def should_run_pms_model(self):
        """
        Flow control for generating PMS classifier model
        """
        if self.args.no_pms_prediction:
            return False
        return True

    def should_generate_pms_uncertainties(self):
        """
        Flow control for generating PMS rergression model uncertainties
        """
        if self.args.pms_uncertainty > 0:
            return True
        return False

    def should_run_age_model(self):
        """
        Flow control for generating age rergression model
        """
        if self.args.no_age_prediction:
            return False
        return True

    def should_generate_age_uncertainties(self):
        """
        Flow control for generating age rergression model uncertainties
        """
        if self.args.age_uncertainty > 0:
            return True
        return False

    def check_valid_args_config(self):
        """
        Makes sure a valid configuration of input arguments was specified
        """
        is_valid = True
        # Uncertainty Misconfigs
        if self.args.av_uncertainty < 0:
            print("ERROR: Bad input argument. Av uncertainty sampling must be greater than 0.",
                    file=sys.stderr)
            is_valid = False
        if self.args.pms_uncertainty < 0:
            print("ERROR: Bad input argument. Yso uncertainty sampling must be greater than 0.",
                    file=sys.stderr)
            is_valid = False
        if self.args.age_uncertainty < 0:
            print("ERROR: Bad input argument. Age uncertainty sampling must be greater than 0.",
                    file=sys.stderr)
            is_valid = False
        if self.args.download_only:
            if self.args.av_uncertainty > 0:
                print("ERROR: Bad input argument. Cannot download only and run av uncertainty.",
                        file=sys.stderr)
                is_valid = False
            if self.args.pms_uncertainty > 0:
                print("ERROR: Bad input argument. Cannot download only and run pms uncertainty.",
                        file=sys.stderr)
                is_valid = False
            if self.args.age_uncertainty > 0:
                print("ERROR: Bad input argument. Cannot download only and run pms uncertainty.",
                        file=sys.stderr)
                is_valid = False
        if self.args.av_scatter_range < 0:
            print("ERROR: Bad input arguments. " +
                        "Av scattering value must be greater than or equal to 0.",
                        file=sys.stderr)
            is_valid = False
        # Batch Size
        if self.args.batch_size <= 0:
            print("ERROR: Bad input argument. Model batch size must be greater than 0.",
                        file=sys.stderr)
            is_valid = False
        # File I/O
        if not os.path.exists(self.args.tableIn) and not self.args.single_object:
            print("ERROR: No file located at \"" + self.args.tableIn + "\"",
                        file=sys.stderr)
            is_valid = False
        elif self.args.single_object and not self.args.tableIn.isdigit():
            print('Cannot use the input as a valid source_id')
            is_valid = False
        
        if not self.args.test:
            if os.path.exists(self.get_output_table_name()):
                print("WARNING: Output will overwrite file at \"" +
                        self.get_output_table_name()+ "\"",
                        file=sys.stderr)
        output_column_names = [self.args.av_out, self.args.pms_out, self.args.age_out]
        for idx, col1 in enumerate(output_column_names[:-1]):
            for col2 in output_column_names[idx+1:]:
                if col1 == col2:
                    print("ERROR: The user specified output columns " +
                        "{} and {} cannot be the same.".format(col1,col2),
                        file=sys.stderr)
                    is_valid = False
        # Control Flow
        if self.args.no_av_prediction and not self.args.av:
            if self.args.pms_uncertainty or self.args.age_uncertainty:
                print("ERROR: To use the generate PMS and/or age model uncertainties " +
                        "the pipeline must either generate Av values or Av values must " +
                        "be specified with the --av flag.",
                        file=sys.stderr)
                is_valid = False
            if not self.args.no_pms_prediction or not self.args.no_age_prediction:
                print("ERROR: To use the PMS and/or age models the pipeline must either " +
                        "generate Av values or Av values must be specified with the --av flag.",
                        file=sys.stderr)
                is_valid = False
        return is_valid

#--------------- Command Line Argument Parser ---------------#
def parse_args():
    """
    Pipeline command line arguments
    """
    parser = argparse.ArgumentParser()
    # Main Pipeline Control Options:
    sagitta_options = parser.add_argument_group("Sagitta pipeline options")
    sagitta_options.add_argument("tableIn",
                        help="File path and name for table with Gaia source ids (.fits file) OR" +
                                "a single source_id (if specified as --single_object)",
                        type=str)
    sagitta_options.add_argument("--tableOut",
                        help="File path and name for where to save the output table " +
                                "(.fits file) [default: \"{tableIn}-sagitta.fits\"]",
                        type=str)
    sagitta_options.add_argument("--download_only",
                        help="Will only download the missing/unspecified Gaia/2MASS data and" +
                                "will not run any of the models " +
                                "(i.e. no Av, PMS or age predicitons).",
                        action="store_true")
    sagitta_options.add_argument("--version",
                        help="Gaia data release version to download data" +
                                "[default: \"edr3\", allows \"dr2\"|\"edr3\"]",
                        default="edr3",
                        type=str)
    #sagitta_options.add_argument("--ignore_bad_rows",
    #                    help="Will suppress a check for rows that have missing or duplicate source_id",
    #                    action="store_true")
    sagitta_options.add_argument("--single_object",
                        help="Instead of reading in a table, will create one using a single source_id",
                        action="store_true")
    sagitta_options.add_argument("--av",
                        help="The input table column name for stellar extinction " +
                                "(case insensitive) [default: Off]" +
                                "[NOTE: If specified, this column's values will be used in " +
                                "the PMS and age predicitons insead of values generated from " +
                                "the Av model. Thus, this option should only be used to " +
                                "prevent repetitive use of the extinction model or to " +
                                "directly see how the models respond to differing Av inputs.]",
                        type=str)
    sagitta_options.add_argument("--no_av_prediction",
                        help="Will not run the stellar extinction prediction model",
                        action="store_true")
    sagitta_options.add_argument("--no_pms_prediction",
                        help="Will not run the pms probablilty prediction model",
                        action="store_true")
    sagitta_options.add_argument("--no_age_prediction",
                        help="Will not run the stellar age prediction model",
                        action="store_true")
    sagitta_options.add_argument("--test",
                        help="Used for testing. Only runs the first 10,000 stars in the input " +
                                "table and and saves the output with a \"test\" label.",
                        action="store_true")
    uncertainty_options = parser.add_argument_group("Prediction uncertinty generation options")
    uncertainty_options.add_argument("--av_uncertainty",
                        help="Calculate extinction model output uncertainty values for each " +
                                "star by sampling each star this many times [default: 0 (off)]",
                        default=0,
                        type=int)
    uncertainty_options.add_argument("--pms_uncertainty",
                        help="Calculate PMS model output uncertainty values for each star " +
                                "by sampling each star this many times [default: 0 (off)]",
                        default=0,
                        type=int)
    uncertainty_options.add_argument("--age_uncertainty",
                        help="Calculate age model output uncertainty values for each star " +
                                "by sampling each star this many times [default: 0 (off)]",
                        default=0,
                        type=int)
    uncertainty_options.add_argument("--av_scatter_range",
                        help="When generating the PMS and age model output uncertainties the " +
                                "extinction (Av) column values will be scattered by +/- this " +
                                "much [default: 0.1]",
                        default=0.1,
                        type=int)
    # Processing Options:
    processing_options = parser.add_argument_group("Processing options")
    processing_options.add_argument("--batch_size",
                        help="The batch size for making predictions [default: 5000]",
                        default=5000,
                        type=int)
    processing_options.add_argument("--device",
                        help="The torch device to use for running predictions [default: \"cpu\"]",
                        default="cpu",
                        type=str)
    # Position and Source ID Naming Options:
    identifying_column_names = parser.add_argument_group("Input stellar identification " +
                                        "renaming options [Note: Use just l & b or ra & dec]")
    identifying_column_names.add_argument("--source_id",
                        help="The input table column name containing star Gaia IDs " +
                                "(case insensitive) [default: \"source_id\"]",
                        default="source_id",
                        type=str)
    identifying_column_names.add_argument("--l",
                        help="The input table column name containing Glactic Longitude " +
                                "(case insensitive) [default: \"l\"]",
                        default="l",
                        type=str)
    identifying_column_names.add_argument("--b",
                        help="The input table column name containing Glactic Latitude " +
                                "(case insensitive) [default: \"b\"]",
                        default="b",
                        type=str)
    identifying_column_names.add_argument("--ra",
                        help="The input table column name containing Right Ascension " +
                                "(case insensitive) [default: \"ra\"]",
                        default="ra",
                        type=str)
    identifying_column_names.add_argument("--dec",
                        help="The input table column name containing Declination " +
                                "(case insensitive) [default: \"dec\"]",
                        default="dec",
                        type=str)
    # Photometric Field Naming Options:
    data_column_names = parser.add_argument_group("Input stellar parameter renaming options")
    data_column_names.add_argument("--g",
                        help="The input table column name containing G flux magnitude " +
                                "(case insensitive) [default: \"g\"]",
                        default="g",
                        type=str)
    data_column_names.add_argument("--bp",
                        help="The input table column name containing BP flux magnitude " +
                                "(case insensitive) [default: \"bp\"]",
                        default="bp",
                        type=str)
    data_column_names.add_argument("--rp",
                        help="The input table column name containing RP flux magnitude " +
                                "(case insensitive) [default: \"rp\"]",
                        default="rp",
                        type=str)
    data_column_names.add_argument("--j",
                        help="The input table column name containing J flux magnitude " +
                                "(case insensitive) [default: \"j\"]",
                        default="j",
                        type=str)
    data_column_names.add_argument("--h",
                        help="The input table column name containing H flux magnitude " +
                                "(case insensitive) [default: \"h\"]",
                        default="h",
                        type=str)
    data_column_names.add_argument("--k",
                        help="The input table column name containing K flux magnitude " +
                                "(case insensitive) [default: \"k\"]",
                        default="k",
                        type=str)
    data_column_names.add_argument("--parallax",
                        help="The input table column name containing Parallax " +
                                "(case insensitive) [default: \"parallax\"]",
                        default="parallax",
                        type=str)
    # Photometric Error Field Naming Options:
    error_column_names = parser.add_argument_group("Input stellar errors renaming options")
    error_column_names.add_argument("--eg",
                        help="The input table column name containing G flux magnitude error " +
                                "(case insensitive) [default: \"eg\"]",
                        default="eg",
                        type=str)
    error_column_names.add_argument("--ebp",
                        help="The input table column name containing BP flux magnitude error " +
                                "(case insensitive) [default: \"ebp\"]",
                        default="ebp")
    error_column_names.add_argument("--erp",
                        help="The input table column name containing RP flux magnitude error " +
                                "(case insensitive) [default: \"erp\"]",
                        default="erp",
                        type=str)
    error_column_names.add_argument("--ej",
                        help="The input table column name containing J flux magnitude error " +
                                "(case insensitive) [default: \"ej\"]",
                        default="ej",
                        type=str)
    error_column_names.add_argument("--eh",
                        help="The input table column name containing H flux magnitude error " +
                                "(case insensitive) [default: \"eh\"]",
                        default="eh",
                        type=str)
    error_column_names.add_argument("--ek",
                        help="The input table column name containing K flux magnitude error " +
                                "(case insensitive) [default: \"ek\"]",
                        default="ek",
                        type=str)
    error_column_names.add_argument("--eparallax",
                        help="The input table column name containing parallax error " +
                                "(case insensitive) [default: \"eparallax\"]",
                        default="eparallax",
                        type=str)
    # Model Output Field Naming Options:
    output_column_names = parser.add_argument_group("Output coulumn renaming options " +
                                    "[Note: This will set names for uncertainty outputs too]")
    output_column_names.add_argument("--av_out",
                        help="The column name that will hold the predicted extinction output " +
                                "[default: \"av\"]",
                        default="av",
                        type=str)
    output_column_names.add_argument("--pms_out",
                        help="The column name that will hold the predicted pms probablility " +
                                "output [default: \"pms\"]",
                        default="pms",
                        type=str)
    output_column_names.add_argument("--age_out",
                        help="The column name that will hold the predicted star ages " +
                                "[default: \"age\"]",
                        default="age",
                        type=str)
    return parser.parse_args()

#--------------- Main Function Catch ---------------#
if __name__ == "__main__":
    main()
