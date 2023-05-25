import os.path

import pandas as pd
import motmetrics as mm

# 读入 predict track 和 ground truth csv 文件
# predict_file = r"G:\20x_dataset\evaluate_data\split-copy19\group0\track-GT.csv"
# ground_truth_file = r"G:\20x_dataset\evaluate_data\split-copy19\group0\tracking_output\track.csv"

# predict_file = r"G:\20x_dataset\evaluate_data\src06\trackmeta.csv"
# predict_file = r"G:\20x_dataset\evaluate_data\src06\track\track.csv"
# predict_file = r"E:\paper\evaluate_data\src06\tracking_output\track.csv"
# ground_truth_file = r"E:\paper\evaluate_data\src06\track-GT.csv"
#
# # 从CSV文件中读取跟踪和真实轨迹
# predict_df = pd.read_csv(predict_file)
# truth_df = pd.read_csv(ground_truth_file)
#
# predict_df = predict_df.sort_values(by=['track_id', 'cell_id', 'frame_index'])
#
# truth_df = truth_df.sort_values(by=['track_id', 'cell_id', 'frame_index'])


def prepare_data(predict_file_path, ground_truth_file_path):
    predict_df = pd.read_csv(predict_file_path)
    truth_df = pd.read_csv(ground_truth_file_path)
    try:
        predict_df = predict_df.sort_values(by=['track_id', 'cell_id', 'frame_index'])
    except KeyError:
        predict_df = predict_df.sort_values(by=['cell_id', 'frame_index'])
    truth_df = truth_df.sort_values(by=['track_id', 'cell_id', 'frame_index'])
    return truth_df, predict_df


def evaluate(truth_df, predict_df, outfile=None):
    # Create an accumulator that will be used to accumulate errors.
    # It accepts two arguments: the list of metric names to compute, and whether the metrics are "single object" metrics
    # (meaning they're computed on a per-object basis, like tracking accuracy or recall), or "tracking metrics" (which consider
    # the tracking as a whole and measure things like identity switches or fragmentation)
    # metric_names = ['recall', 'precision', 'num_false_positives', 'num_misses', 'mota', 'motp', 'idf1']
    metric_names = ['num_frames', 'idf1', 'idp', 'idr',
                    'recall', 'precision', 'num_objects',
                    'mostly_tracked', 'partially_tracked',
                    'mostly_lost', 'num_false_positives',
                    'num_misses', 'num_switches',
                    'num_fragmentations', 'mota',  'num_false_positives',  'num_misses'
                    ]

    acc = mm.MOTAccumulator()

    # Update the accumulator for each frame in the sequence. In this example we assume the two dataframes (gt and dt)
    # have the same number of rows and are in the same order. If this is not the case you will need to perform some
    # sort of alignment of the dataframes (for example, sorting them by frame index and track ID) before calling update.
    for i, (frame_gt, frame_dt) in enumerate(zip(truth_df.groupby('frame_index'), predict_df.groupby('frame_index'))):
        _, gt_group = frame_gt
        _, dt_group = frame_dt

        # The update() function takes four arrays:
        # - oids (the list of IDs of ground truth objects present in the frame)
        # - hids (the list of IDs of detected objects present in the frame)
        # - dists (a 2D array with shape [len(oids), len(hids)] containing the pairwise distances between the ground truth and
        #         detected objects)
        # - frameid (an optional frame ID that can be used to identify the frame in case the dataframes aren't sorted by frame)
        # In this example we assume that the track IDs in the ground truth and detection dataframes are the same.
        # If the track IDs don't match, you will need to perform some kind of matching or linking step before calling update().
        oids = gt_group['cell_id'].values
        hids = dt_group['cell_id'].values
        dists = mm.distances.norm2squared_matrix(gt_group[['center_x', 'center_y']].values,
                                                 dt_group[['center_x', 'center_y']].values)
        try:
            acc.update(oids, hids, dists, frameid=i)
        except KeyError:
            continue

    # Compute metrics on the accumulated errors.
    # The compute_metrics() function returns a dictionary with the computed metrics.
    # The first argument is the accumulator to compute the metrics on. The second argument is the metric to compute. It
    # can be a string (e.g. "mota") or a list of strings (e.g. ["mota", "num_false_positives"]).
    metrics = mm.metrics.create()
    summary = metrics.compute(acc, metrics=metric_names)
    # summary = metrics.compute(acc, metrics=list(mm.metrics.motchallenge_metrics))

    strsummary = mm.io.render_summary(
        summary,
        # formatters={'mota' : '{:.2%}'.format},
        namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll',
                 'precision': 'Prcn', 'num_objects': 'GT',
                 'mostly_tracked': 'MT', 'partially_tracked': 'PT',
                 'mostly_lost': 'ML', 'num_false_positives': 'FP',
                 'num_misses': 'FN', 'num_switches': 'IDsw',
                 'num_fragmentations': 'FM', 'mota': 'MOTA', 'motp': 'MOTP',
                 }
    )
    print(mm.io.render_summary(summary, formatters=metrics.formatters, namemap=mm.io.motchallenge_metric_names))
    if outfile:
        summary.to_csv(outfile)
    return summary

def run_evaluate(gap):

    # dirs = ['copy_of_1_xy01', 'copy_of_1_xy19', 'MCF10A_copy02', 'MCF10A_copy11', 'src06']
    dirs = ['src06']
    base = r'G:\paper\evaluate_data'
    for i in dirs:
        prediction_CCDeep = rf'G:\paper\evaluate_data\{gap*5}min\{i}_{gap*5}min\tracking_output\track.csv'
        prediction_pcnadeep = rf'G:\paper\evaluate_data\{gap*5}min\{i}_{gap*5}min\track\refined-pcnadeep(CCDeep_format).csv'
        prediction_GT = rf"G:\paper\evaluate_data\{gap*5}min\{i}_{gap*5}min\{gap*5}-track-GT.csv"
        prediction_trackmeta = rf"G:\paper\evaluate_data\{gap*5}min\{i}_{gap*5}min\export.csv"
        out = rf"G:\paper\evaluate_data\{gap*5}min\{i}_{gap*5}min\evaluate-track.csv"
        print(prediction_GT)
        print(prediction_CCDeep)
        print(prediction_pcnadeep)
        print(out)


        ccdeep_result = evaluate(*prepare_data(prediction_CCDeep, prediction_GT))
        pcnadeep_result = evaluate(*prepare_data(prediction_pcnadeep, prediction_GT))
        trackmeta_result = evaluate(*prepare_data(prediction_trackmeta, prediction_GT))
        result = pd.concat([ccdeep_result, pcnadeep_result, trackmeta_result])
        result.index = ['CCDeep', 'pcnadeep', 'trackmeta']
        print(result)
        result.to_csv(out, index=True)

    # pred = rf'E:\paper\evaluate_data\{gap*5}min\copy_of_1_xy01_{gap*5}min\tracking_output\(new)track.csv'
    # pred2 = rf'E:\paper\evaluate_data\{gap*5}min\copy_of_1_xy01_{gap*5}min\track\refined-pcnadeep(CCDeep_format).csv'
    # gt = rf'E:\paper\evaluate_data\{gap*5}min\copy_of_1_xy01_{gap*5}min\{gap*5}-track-GT.csv'
    # out = rf'E:\paper\evaluate_data\{gap*5}min\copy_of_1_xy01_{gap*5}min\evaluate-track.csv'
    # evaluate(*prepare_data(pred, gt), outfile=out)
    # evaluate(*prepare_data(pred2, gt), outfile=out)

def run_without_trackmeta(gap):

    dirs = ['copy_of_1_xy01', 'copy_of_1_xy19', 'MCF10A_copy02', 'MCF10A_copy11', 'src06']
    # dirs = ['src06']
    base = r'G:\paper\evaluate_data'
    for i in dirs:
        prediction_CCDeep = rf'G:\paper\evaluate_data\{gap*5}min\{i}_{gap*5}min\tracking_output\track.csv'
        prediction_pcnadeep = rf'G:\paper\evaluate_data\{gap*5}min\{i}_{gap*5}min\track\refined-pcnadeep(CCDeep_format).csv'
        prediction_GT = rf"G:\paper\evaluate_data\{gap*5}min\{i}_{gap*5}min\{gap*5}-track-GT.csv"
        # prediction_trackmeta = rf"G:\paper\evaluate_data\{gap*5}min\{i}_{gap*5}min\export.csv"
        out = rf"G:\paper\evaluate_data\{gap*5}min\{i}_{gap*5}min\evaluate-track.csv"
        print(prediction_GT)
        print(prediction_CCDeep)
        print(prediction_pcnadeep)
        print(out)
        ccdeep_result = evaluate(*prepare_data(prediction_CCDeep, prediction_GT))
        pcnadeep_result = evaluate(*prepare_data(prediction_pcnadeep, prediction_GT))
        # trackmeta_result = evaluate(*prepare_data(prediction_trackmeta, prediction_GT))
        result = pd.concat([ccdeep_result, pcnadeep_result])
        result.index = ['CCDeep', 'pcnadeep']
        print(result)
        result.to_csv(out, index=True)


def evaluate_loss_detection():
    gap = 1
    dirs = ['copy_of_1_xy01',  'copy_of_1_xy19']
    # dirs = ['copy_of_1_xy01', 'copy_of_1_xy19', 'MCF10A_copy02', 'MCF10A_copy11', 'src06']
    for i in dirs:
        ratio = [0.05, 0.1, 0.2, 0.3, 0.5]
        for r in ratio:
            prediction_CCDeep_loss = rf'G:\paper\evaluate_data\{gap * 5}min\{i}_{gap * 5}min\detection_loss_test\{int(r * 100)}%\tracking_output\track.csv'
            prediction_pcnadeep_loss = rf'G:\paper\evaluate_data\{gap * 5}min\{i}_{gap * 5}min\detection_loss_test\{int(r * 100)}%\track\refined-pcnadeep(CCDeep_format).csv'
            prediction_GT = rf"G:\paper\evaluate_data\{gap*5}min\{i}_{gap*5}min\{gap*5}-track-GT.csv"

            out = rf'G:\paper\evaluate_data\{gap * 5}min\{i}_{gap * 5}min\detection_loss_test\{int(r * 100)}%\evaluate-loss-track.csv'
            # ccdeep_result = evaluate(*prepare_data(prediction_CCDeep, prediction_GT))
            ccdeep_loss_result = evaluate(*prepare_data(prediction_CCDeep_loss, prediction_GT))

            # pcnadeep_result = evaluate(*prepare_data(prediction_pcnadeep, prediction_GT))
            pcnadeep_loss_result = evaluate(*prepare_data(prediction_pcnadeep_loss, prediction_GT))

            print(ccdeep_loss_result)
            print(pcnadeep_loss_result)

            result = pd.concat([ccdeep_loss_result, pcnadeep_loss_result])
            result.index = ['CCDeep', 'pcnadeep']
            print(result)
            result.to_csv(out, index=True)

if __name__ == '__main__':
    evaluate_loss_detection()
    # run_without_trackmeta(6)
    # run_evaluate(2)
    # gap = 6
    # pred = rf'E:\paper\evaluate_data\{gap*5}min\copy_of_1_xy01_{gap*5}min\tracking_output\(new)track.csv'
    # pred2 = rf'E:\paper\evaluate_data\{gap*5}min\copy_of_1_xy01_{gap*5}min\track\refined-pcnadeep(CCDeep_format).csv'
    # gt = rf'E:\paper\evaluate_data\{gap*5}min\copy_of_1_xy01_{gap*5}min\{gap*5}-track-GT.csv'
    # out = rf'E:\paper\evaluate_data\{gap*5}min\copy_of_1_xy01_{gap*5}min\evaluate-track.csv'
    # evaluate(*prepare_data(pred, gt), outfile=out)
    # evaluate(*prepare_data(pred2, gt), outfile=out)

    # evaluate(*prepare_data(r'E:\paper\evaluate_data\copy_of_1_xy01\tracking_output\track.csv',
    #                        r'E:\paper\evaluate_data\copy_of_1_xy01\track-GT.csv'), outfile=r'E:\paper\evaluate_data\copy_of_1_xy01\evaluate.csv')


