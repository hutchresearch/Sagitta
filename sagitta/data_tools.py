"""
Data tools for the Sagitta young star photometric classification and age regression pipeline
Paper: Untangling the Galaxy III: Photometric Search for Pre-main Sequence Stars with Deep Learning
Authors: Aidan McBride, Ryan Lingg, Marina Kounkel, Kevin Covey, and Brian Hutchinson
"""

#--------------- External Imports ---------------#
import sys
import numpy as np
import torch
import torch.utils.data
from astropy.table import Table

#--------------- Data Tools ---------------#
class DataTools():
    """
    Class to hold all of the relevant data tool static functions
    """

    @staticmethod
    def normalize_gaia(column_vals, column_name, back=False):
        """
        Scales the values of the array into the into +/- 1 for the specific column (Back == False)
        or takes the normalized values and returns them to their initial values (Back == True)
        """
        sensor_ranges = {
            "g":  [4.0, 20.0],
            "bp": [0.0, 21.0],
            "rp": [0.0, 19.0],
            "j":  [0.0, 18.0],
            "h":  [0.0, 18.0],
            "k":  [0.0, 18.0],
            "parallax": [0.0, 29.0],
            "av": [0.0, 5.0],
            'age': [6.0 , 8.0]
        }
        output = None
        normalized_range = [-1, 1]
        for sensor_name in sensor_ranges:
            range_vals = sensor_ranges[sensor_name]
            if column_name == sensor_name:
                output = DataTools.normalize_gaia_helper(
                                        column_vals,
                                        normalized_range,
                                        range_vals,
                                        back
                                        )
        return output

    @staticmethod
    def normalize_gaia_helper(column_vals, normalized_range, range_vals, back):
        """
        Does the normalization calculation for normalize_gaia
        """
        if back:
            output = (
                    ((column_vals-normalized_range[0])/(normalized_range[1]-normalized_range[0]))*
                    (range_vals[1]-range_vals[0])
                ) + range_vals[0]
        else:
            output = (
                    ((column_vals-range_vals[0])/(range_vals[1]-range_vals[0]))*
                    (normalized_range[1]-normalized_range[0])
                ) + normalized_range[0]
        return output

    @staticmethod
    def astropy_from_table(data_frame):
        """
        1) Converts the column names of the fits file to lowercase
        2) Renames columns based on the contents of column_renaming
        """
        
        for col in data_frame.keys():
            data_frame.rename_column(col,col.lower())
            col=col.lower()
        return data_frame

    @staticmethod
    def download_missing_fields(data_frame, missing_fields,ver):
        """
        1) Downloads any missing data columns as specified in missing_fields
        2) Returns a table with only the downloaded fields plus the source ids
        """
        from astroquery.gaia import Gaia
        print("Downloading missing fields:", ", ".join(missing_fields))
        missing_name_to_query = {
            "l"         : "g.l",
            "b"         : "g.b",
            "pmra"      : "g.pmra",
            "pmdec"     : "g.pmdec",
            "epmra"     : "g.pmra_error as epmra",
            "epmdec"    : "g.pmdec_error as epmdec",
            "parallax"  : "g.parallax",
            "eparallax" : "g.parallax_error as eparallax",
            "g"         : "g.phot_g_mean_mag as g",
            "eg"        : "-2.5*log10(abs(phot_g_mean_flux-phot_g_mean_flux_error))\
                        +2.5*log10(abs(phot_g_mean_flux+phot_g_mean_flux_error)) as eg",
            "bp"        : "g.phot_bp_mean_mag as bp",
            "ebp"       : "-2.5*log10(abs(phot_bp_mean_flux-phot_bp_mean_flux_error))\
                        +2.5*log10(abs(phot_bp_mean_flux+phot_bp_mean_flux_error)) as ebp",
            "rp"        : "g.phot_rp_mean_mag as rp",
            "erp"       : "-2.5*log10(abs(phot_rp_mean_flux-phot_rp_mean_flux_error))\
                        +2.5*log10(abs(phot_rp_mean_flux+phot_rp_mean_flux_error)) as erp",
            "j"         : "tm.j_m as j",
            "ej"        : "tm.j_msigcom as ej",
            "h"         : "tm.h_m as h",
            "eh"        : "tm.h_msigcom as eh",
            "k"         : "tm.ks_m as k",
            "ek"        : "tm.ks_msigcom as ek"
        }
        if "source_id" in data_frame.columns:
            input_table = Table([data_frame["source_id"]], names=["source_id"])
            fields_to_download = ""
            for field in missing_fields:
                fields_to_download = fields_to_download + ", " + missing_name_to_query[field]
                
            if ver=='dr2':
                print('Using Gaia DR2')
                query_string = "SELECT g.source_id" + fields_to_download + " \
                            FROM gaiadr2.gaia_source AS g \
                            inner join TAP_UPLOAD.input_table AS input_table \
                            ON g.source_id = input_table.source_id \
                            LEFT OUTER JOIN gaiadr2.tmass_best_neighbour AS xmatch \
                            ON g.source_id = xmatch.source_id \
                            LEFT OUTER JOIN gaiadr1.tmass_original_valid AS tm \
                            ON tm.tmass_oid = xmatch.tmass_oid"
            elif ver=='edr3':
                print('Using Gaia EDR3')
                query_string = "SELECT g.source_id" + fields_to_download + " \
                            FROM gaiaedr3.gaia_source AS g \
                            inner join TAP_UPLOAD.input_table AS input_table \
                            ON g.source_id = input_table.source_id \
                            LEFT OUTER JOIN gaiaedr3.tmass_psc_xsc_best_neighbour AS xmatch \
                            ON g.source_id = xmatch.source_id \
                            LEFT OUTER JOIN gaiaedr3.tmass_psc_xsc_join AS xmatch_join \
                            ON xmatch.clean_tmass_psc_xsc_oid = xmatch_join.clean_tmass_psc_xsc_oid \
                            LEFT OUTER JOIN gaiadr1.tmass_original_valid AS tm \
                            ON tm.designation = xmatch_join.original_psc_source_id"
            else:
                print('Unsupported Gaia data release. Cannot download photometry, aborting.')
                sys.exit()
                
            query_result = Gaia.launch_job_async(
                query = " ".join(query_string.split()),
                upload_resource = input_table,
                upload_table_name = "input_table",
                verbose = False)
            missing_fields_table = query_result.get_results()
            if len(missing_fields_table)==0:
                print('Could not query any data for the specified source_id')
                sys.exit()
            return missing_fields_table
        print("Error: The input table MUST contain a \"source_id\" column.", file=sys.stderr)
        sys.exit()

    @staticmethod
    def fix_and_mark_nans(data_frame, nan_column_extension='_nan_mask'):
        """
        1) Replaces the dataframe's nan values with boundry values
        2) Appends a boolean mask to the dataframe where values were filled in
            2.1) nan_column_extension is the suffix used in the nan mask columns
        """
        nan_replacements = {
            "g"         : 20,
            "bp"        : 21,
            "rp"        : 19,
            "j"         : 18,
            "h"         : 18,
            "k"         : 18,
            "parallax"  : 0,
            "av"        : 5,
            "eg"        : 0.1,
            "ebp"       : 0.1,
            "erp"       : 0.1,
            "ej"        : 0.1,
            "eh"        : 0.1,
            "ek"        : 0.1,
            "eparallax" : 2
        }
        for _, (column, nan_replacement) in enumerate(nan_replacements.items()):
            if column in data_frame.columns:
                try:
                    mask_col_name = column + nan_column_extension
                    data_frame[mask_col_name] = np.ma.getmask(np.ma.masked_invalid(data_frame[column]))
                    if data_frame[mask_col_name].any(): data_frame[column] = data_frame[column].filled(nan_replacement)
                except:
                    mask_col_name = column + nan_column_extension
                    data_frame[mask_col_name] = np.isnan(data_frame[column])
                    data_frame[column][np.isnan(data_frame[column])] = nan_replacement
        return data_frame

class SagittaDataset(torch.utils.data.Dataset):
    """
    This class is an extension of the pytorch dataset
    It is responsable for feeding the data into each of the models
    Functionality:
        1) Extracts only the columns needed for loading data into the desired model
        2) Performs normalization on the values
        3) Formats the input as numpy arrays
    """
    def __init__(self, frame, data_format, column_names):
        if data_format == "StellarExtinction":
            av_input_names = [column_names[std_name] for std_name in ["l", "b", "parallax"]]
            av_input = frame[av_input_names]
            av_input[column_names["l"]] = av_input[column_names["l"]]/360
            av_input[column_names["b"]] = (av_input[column_names["b"]]/180)+0.5
            av_input[column_names["parallax"]] = av_input[column_names["parallax"]]/5
            av_input = np.lib.recfunctions.structured_to_unstructured(av_input.as_array()).astype(np.float32)
            if np.ma.isMaskedArray(av_input): av_input=av_input.filled()
            self.inputs = np.reshape(av_input, (-1, 1, 3))
        elif data_format == "PMSClassifier":
            pms_std_input_names = ["parallax", "av", "g", "bp", "rp", "j", "h", "k"]
            pms_input_names = [column_names[std_name] for std_name in pms_std_input_names]
            pms_input = frame[pms_input_names]
            std_input_column_names = {v: k for k, v in column_names.items()}
            for col in pms_input.columns:
                pms_input[col] = DataTools.normalize_gaia(
                                            column_vals=pms_input[col],
                                            column_name=std_input_column_names[col]
                                            )
            pms_input = np.lib.recfunctions.structured_to_unstructured(pms_input.as_array()).astype(np.float32)
            if np.ma.isMaskedArray(pms_input): pms_input=pms_input.filled()
            self.inputs = np.reshape(pms_input, (-1, 1, 8))
        elif data_format == "YoungStarAgeRegressor":
            age_std_input_names = ["parallax", "av", "g", "bp", "rp", "j", "h", "k"]
            age_input_names = [column_names[std_name] for std_name in age_std_input_names]
            age_input = frame[age_input_names]
            std_input_column_names = {v: k for k, v in column_names.items()}
            for col in age_input.columns:
                age_input[col] = DataTools.normalize_gaia(
                                            column_vals=age_input[col],
                                            column_name=std_input_column_names[col]
                                            )
            age_input = np.lib.recfunctions.structured_to_unstructured(age_input.as_array()).astype(np.float32)
            if np.ma.isMaskedArray(age_input): age_input=age_input.filled()
            self.inputs = np.reshape(age_input, (-1, 1, 8))
        else:
            print("Unknown dataset format specified: {}".format(data_format), file=sys.stderr)
            sys.exit()

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, index):
        return self.inputs[index]
