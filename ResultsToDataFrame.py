import numpy as np
import pandas as pd

df = pd.read_csv('TestSet/Results.csv',
                 index_col=False)



def record_to_df(i, face_mask_decision,
                 face_mask_mouth_mask_results,
                 face_mask_image_class_results,
                 face_mask_gray_scaling_results):
    i -= 1

    df.loc[i, 'Ensemble'] = face_mask_decision

    df.loc[i, 'Mouth Detection'] = face_mask_mouth_mask_results

    df.loc[i, 'Image Classification'] = face_mask_image_class_results

    df.loc[i, 'Greyscale'] = face_mask_gray_scaling_results

    df['Gap'] = ""
    df['Gap1'] = ""

    df['Ensemble Confusion Matrix 1 (Simple)'] = confusion_matrix_one(df[['Correct', 'Ensemble']])
    df["Ensemble Summary One"] = sum_of_matrix(df['Ensemble Confusion Matrix 1 (Simple)'])
    df["Ensemble Evaluation One"] = matrix_accuracy(df["Ensemble Summary One"])
    df['Ensemble Space'] = ""

    df['Mouth Detection Matrix 1 (Simple)'] = confusion_matrix_one(df[['Correct', 'Mouth Detection']])
    df["Mouth Detection Summary One"] = sum_of_matrix(df['Mouth Detection Matrix 1 (Simple)'])
    df["Mouth Detection Evaluation One"] = matrix_accuracy(df["Mouth Detection Summary One"])
    df['Mouth Space'] = ""

    df['Image Classification Matrix 1 (Simple)'] = confusion_matrix_one(df[['Correct', 'Image Classification']])
    df["Image Classification Summary One"] = sum_of_matrix(df['Image Classification Matrix 1 (Simple)'])
    df["Image Classification Evaluation One"] = matrix_accuracy(df["Image Classification Summary One"])
    df['Image Space'] = ""

    df['Greyscale Confusion Matrix 1 (Simple)'] = confusion_matrix_one(df[['Correct', 'Greyscale']])
    df["Greyscale Summary One"] = sum_of_matrix(df['Greyscale Confusion Matrix 1 (Simple)'])
    df["Greyscale Evaluation One"] = matrix_accuracy(df["Greyscale Summary One"])

    df.to_csv("Results/Output.csv")



def confusion_matrix_one(dataframe):
    frame = pd.DataFrame(dataframe)

    frame['Confusion Matrix'] = np.where(
        np.logical_and(frame.iloc[:, 0] > 0, frame.iloc[:, 1] >= 1),
        "TP",
        np.where(
            np.logical_and(frame.iloc[:, 0] == 0, frame.iloc[:, 1] >= 1),
            "FP",
            np.where(np.logical_and(frame.iloc[:, 0] == 0, frame.iloc[:, 1] == 0),
                     "TN",
                     np.where(np.logical_and(frame.iloc[:, 0] > 0, frame.iloc[:, 1] == 0),
                              "FN",
                              "N/A"))))

    return frame['Confusion Matrix']


def sum_of_matrix(dataframe):
    frame = pd.DataFrame(dataframe)

    frame.loc[0, "Summary"] = "True Positives sum"
    frame.loc[1, "Summary"] = np.where(frame.iloc[:, 0] == "TP", 1, 0).sum()

    frame.loc[2, "Summary"] = "True Negatives sum"
    frame.loc[3, "Summary"] = np.where(frame.iloc[:, 0] == "TN", 1, 0).sum()

    frame.loc[4, "Summary"] = "False Positives sum"
    frame.loc[5, "Summary"] = np.where(frame.iloc[:, 0] == "FP", 1, 0).sum()

    frame.loc[6, "Summary"] = "False Negatives sum"
    frame.loc[7, "Summary"] = np.where(frame.iloc[:, 0] == "FN", 1, 0).sum()

    return frame["Summary"]


def matrix_accuracy(dataframe):
    frame = pd.DataFrame(dataframe)

    frame.loc[0, "Evaluation"] = "Accuracy:"
    frame.loc[1, "Evaluation"] = (frame.iloc[1, 0] + frame.iloc[3, 0]) / (
                frame.iloc[1, 0] + frame.iloc[3, 0] + frame.iloc[5, 0] + frame.iloc[7, 0])

    frame.loc[2, "Evaluation"] = "Error Rate:"
    frame.loc[3, "Evaluation"] = (frame.iloc[5, 0] + frame.iloc[7, 0]) / (
                frame.iloc[1, 0] + frame.iloc[3, 0] + frame.iloc[5, 0] + frame.iloc[7, 0])

    frame.loc[4, "Evaluation"] = "Sensitivity:"
    frame.loc[5, "Evaluation"] = frame.iloc[1, 0] / (
            frame.iloc[1, 0] + frame.iloc[7, 0])


    return frame["Evaluation"]


def confusion_matrix_two(dataframe):

    frame = pd.DataFrame(dataframe)

    frame['Confusion Matrix 2'] = np.where(
        np.logical_and(frame.iloc[:, 0] > 0, frame.iloc[:, 1] == 1),
        "TP",
        np.where(
            np.logical_and(frame.iloc[:, 0] == 0, frame.iloc[:, 1] == 1),
            "FP",
            np.where(np.logical_and(frame.iloc[:, 0] == 0, frame.iloc[:, 1] == 0),
                     "TN",
                     np.where(np.logical_and(frame.iloc[:, 0] > 0, frame.iloc[:, 1] == 0),
                              "FN",
                              "N/A"))))


    return dataframe['Confusion Matrix 2']