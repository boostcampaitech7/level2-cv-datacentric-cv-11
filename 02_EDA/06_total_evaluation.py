# %%
import os
import json
import numpy as np
from pathlib import Path
from collections import namedtuple
from copy import deepcopy
import math


def read_json(filename):
    with Path(filename).open(encoding='utf8') as handle:
        ann = json.load(handle)
    return ann


# 평가를 수행할 언어 목록
languages = ["chinese", "japanese", "thai", "vietnamese"]

# 결과를 저장할 딕셔너리
overall_results = {}


# %%

# ICDAR2015 official evaluation code를 바탕으로 작성되었습니다. : http://rrc.cvc.uab.es/


def default_evaluation_params():
    """
    default_evaluation_params: Default parameters to use for the validation and evaluation.
    """
    return {
        'AREA_RECALL_CONSTRAINT': 0.8,
        'AREA_PRECISION_CONSTRAINT': 0.4,
        'EV_PARAM_IND_CENTER_DIFF_THR': 1,
        'MTYPE_OO_O': 1.,
        'MTYPE_OM_O': 0.8,
        'MTYPE_OM_M': 1.,
        'GT_SAMPLE_NAME_2_ID': 'gt_img_([0-9]+).txt',
        'DET_SAMPLE_NAME_2_ID': 'res_img_([0-9]+).txt',
        'CRLF': False  # Lines are delimited by Windows CRLF format
    }


def calc_deteval_metrics(pred_bboxes_dict, gt_bboxes_dict, transcriptions_dict=None,
                         eval_hparams=None, bbox_format='rect', verbose=False):
    """
    Args:
    pred_bboxes_dict (dict): 각 샘플에 대한 예측된 bounding box의 딕셔너리.
    gt_bboxes_dict (dict): 각 샘플에 대한 실제 정답(ground truth) bouding box의 딕셔너리.
    transcriptions_dict (dict, 선택적): 각 샘플에 대한 텍스트 전사(transcription)의 딕셔너리.
    eval_hparams (dict, 선택적): 평가 하이퍼파라미터.
    bbox_format (str, 선택적): bounding box의 형식. 기본값은 'rect'.
    verbose (bool, 선택적): True일 경우, 출력에 상세한 평가 로그를 포함.

    Returns:
    dict: 계산된 메트릭과 기타 평가 정보를 포함하는 딕셔너리.

    Note:
    현재는 rect(xmin, ymin, xmax, ymax) 형식의 bounding box만 지원함. 다른 형식(quadrilateral, polygon, etc.)의 데이터가 들어오면 외접하는 rect로 변환해서 이용하고 있음.
    """

    def one_to_one_match(row, col):
        cont = 0
        for j in range(len(recallMat[0])):
            if recallMat[row, j] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and precisionMat[row, j] >= eval_hparams['AREA_PRECISION_CONSTRAINT']:
                cont = cont + 1
        if (cont != 1):
            return False
        cont = 0
        for i in range(len(recallMat)):
            if recallMat[i, col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and precisionMat[i, col] >= eval_hparams['AREA_PRECISION_CONSTRAINT']:
                cont = cont + 1
        if (cont != 1):
            return False

        if recallMat[row, col] >= eval_hparams['AREA_RECALL_CONSTRAINT'] and precisionMat[row, col] >= eval_hparams['AREA_PRECISION_CONSTRAINT']:
            return True
        return False

    def num_overlaps_gt(gtNum):
        cont = 0
        for detNum in range(len(detRects)):
            if detNum not in detDontCareRectsNum:
                if recallMat[gtNum, detNum] > 0:
                    cont = cont + 1
        return cont

    def num_overlaps_det(detNum):
        cont = 0
        for gtNum in range(len(recallMat)):
            if gtNum not in gtDontCareRectsNum:
                if recallMat[gtNum, detNum] > 0:
                    cont = cont + 1
        return cont

    def is_single_overlap(row, col):
        if num_overlaps_gt(row) == 1 and num_overlaps_det(col) == 1:
            return True
        else:
            return False

    def one_to_many_match(gtNum):
        many_sum = 0
        detRects = []
        for detNum in range(len(recallMat[0])):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and detNum not in detDontCareRectsNum:
                if precisionMat[gtNum, detNum] >= eval_hparams['AREA_PRECISION_CONSTRAINT']:
                    many_sum += recallMat[gtNum, detNum]
                    detRects.append(detNum)
        if round(many_sum, 4) >= eval_hparams['AREA_RECALL_CONSTRAINT']:
            return True, detRects
        else:
            return False, []

    def many_to_one_match(detNum):
        many_sum = 0
        gtRects = []
        for gtNum in range(len(recallMat)):
            if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCareRectsNum:
                if recallMat[gtNum, detNum] >= eval_hparams['AREA_RECALL_CONSTRAINT']:
                    many_sum += precisionMat[gtNum, detNum]
                    gtRects.append(gtNum)
        if round(many_sum, 4) >= eval_hparams['AREA_PRECISION_CONSTRAINT']:
            return True, gtRects
        else:
            return False, []

    def area(a, b):
        dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin) + 1
        dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin) + 1
        if (dx >= 0) and (dy >= 0):
            return dx*dy
        else:
            return 0.

    def center(r):
        x = float(r.xmin) + float(r.xmax - r.xmin + 1) / 2.
        y = float(r.ymin) + float(r.ymax - r.ymin + 1) / 2.
        return Point(x, y)

    def point_distance(r1, r2):
        distx = math.fabs(r1.x - r2.x)
        disty = math.fabs(r1.y - r2.y)
        return math.sqrt(distx * distx + disty * disty)

    def center_distance(r1, r2):
        return point_distance(center(r1), center(r2))

    def diag(r):
        w = (r.xmax - r.xmin + 1)
        h = (r.ymax - r.ymin + 1)
        return math.sqrt(h * h + w * w)

    if eval_hparams is None:
        eval_hparams = default_evaluation_params()

    if bbox_format != 'rect':
        raise NotImplementedError

    # bbox가 다른 형식일 경우 rect 형식으로 변환
    _pred_bboxes_dict, _gt_bboxes_dict = deepcopy(pred_bboxes_dict), deepcopy(gt_bboxes_dict)
    pred_bboxes_dict, gt_bboxes_dict = dict(), dict()
    for sample_name, bboxes in _pred_bboxes_dict.items():
        # 이미 rect 형식일 경우 변환하지 않고 그대로 사용
        if len(bboxes) > 0 and np.array(bboxes[0]).ndim == 1 and len(bboxes[0]) == 4:
            pred_bboxes_dict = _pred_bboxes_dict
            break

        pred_bboxes_dict[sample_name] = []
        for bbox in map(np.array, bboxes):
            rect = [bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max(), bbox[:, 1].max()]
            pred_bboxes_dict[sample_name].append(rect)
    for sample_name, bboxes in _gt_bboxes_dict.items():
        if len(bboxes) > 0 and np.array(bboxes[0]).ndim == 1 and len(bboxes[0]) == 4:
            gt_bboxes_dict = _gt_bboxes_dict
            break

        gt_bboxes_dict[sample_name] = []
        for bbox in map(np.array, bboxes):
            rect = [bbox[:, 0].min(), bbox[:, 1].min(), bbox[:, 0].max(), bbox[:, 1].max()]
            gt_bboxes_dict[sample_name].append(rect)

    perSampleMetrics = {}

    methodRecallSum = 0
    methodPrecisionSum = 0

    Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
    Point = namedtuple('Point', 'x y')

    numGt = 0
    numDet = 0

    # ground truth 딕셔너리의 각 샘플에 대해 반복
    for sample_name in gt_bboxes_dict:

        recall = 0
        precision = 0
        hmean = 0
        recallAccum = 0.
        precisionAccum = 0.
        gtRects = []
        detRects = []
        gtPolPoints = []
        detPolPoints = []
        gtDontCareRectsNum = []  # Array of Ground Truth Rectangles' keys marked as don't care
        detDontCareRectsNum = []  # Array of Detected Rectangles' matched with a don't care GT
        pairs = []
        evaluationLog = ""

        recallMat = np.empty([1, 1])
        precisionMat = np.empty([1, 1])

        pointsList = gt_bboxes_dict[sample_name]

        if transcriptions_dict is None:
            transcriptionsList = None
        else:
            transcriptionsList = transcriptions_dict[sample_name]

        for n in range(len(pointsList)):
            points = pointsList[n]
            transcription = transcriptionsList[n] if transcriptionsList else None
            dontCare = transcription == "###" if transcription else False
            gtRect = Rectangle(*points)
            gtRects.append(gtRect)
            gtPolPoints.append(np.array(points).tolist())
            if dontCare:
                gtDontCareRectsNum.append(len(gtRects)-1)

        evaluationLog += "GT rectangles: " + \
            str(len(gtRects)) + (" (" + str(len(gtDontCareRectsNum)) +
                                 " don't care)\n" if len(gtDontCareRectsNum) > 0 else "\n")

        if sample_name in pred_bboxes_dict:
            pointsList = pred_bboxes_dict[sample_name]

            for n in range(len(pointsList)):
                points = pointsList[n]
                detRect = Rectangle(*points)
                detRects.append(detRect)
                detPolPoints.append(np.array(points).tolist())
                if len(gtDontCareRectsNum) > 0:
                    for dontCareRectNum in gtDontCareRectsNum:
                        dontCareRect = gtRects[dontCareRectNum]
                        intersected_area = area(dontCareRect, detRect)
                        rdDimensions = ((detRect.xmax - detRect.xmin+1) * (detRect.ymax - detRect.ymin+1))
                        if (rdDimensions == 0):
                            precision = 0
                        else:
                            precision = intersected_area / rdDimensions
                        if (precision > eval_hparams['AREA_PRECISION_CONSTRAINT']):
                            detDontCareRectsNum.append(len(detRects)-1)
                            break

            evaluationLog += "DET rectangles: " + \
                str(len(detRects)) + (" (" + str(len(detDontCareRectsNum)) +
                                      " don't care)\n" if len(detDontCareRectsNum) > 0 else "\n")

            if len(gtRects) == 0:
                recall = 1
                precision = 0 if len(detRects) > 0 else 1

            # Recall 과 precision 매트릭스를 계산
            if len(detRects) > 0:
                outputShape = [len(gtRects), len(detRects)]
                recallMat = np.empty(outputShape)
                precisionMat = np.empty(outputShape)
                gtRectMat = np.zeros(len(gtRects), np.int8)
                detRectMat = np.zeros(len(detRects), np.int8)
                for gtNum in range(len(gtRects)):
                    for detNum in range(len(detRects)):
                        rG = gtRects[gtNum]
                        rD = detRects[detNum]
                        intersected_area = area(rG, rD)
                        rgDimensions = ((rG.xmax - rG.xmin+1) * (rG.ymax - rG.ymin+1))
                        rdDimensions = ((rD.xmax - rD.xmin+1) * (rD.ymax - rD.ymin+1))
                        recallMat[gtNum, detNum] = 0 if rgDimensions == 0 else intersected_area / rgDimensions
                        precisionMat[gtNum, detNum] = 0 if rdDimensions == 0 else intersected_area / rdDimensions

                # Find one-to-one matches
                evaluationLog += "Find one-to-one matches\n"
                for gtNum in range(len(gtRects)):
                    for detNum in range(len(detRects)):
                        if gtRectMat[gtNum] == 0 and detRectMat[detNum] == 0 and gtNum not in gtDontCareRectsNum and detNum not in detDontCareRectsNum:
                            match = one_to_one_match(gtNum, detNum)
                            if match is True:
                                # in deteval we have to make other validation before mark as one-to-one
                                if is_single_overlap(gtNum, detNum) is True:
                                    rG = gtRects[gtNum]
                                    rD = detRects[detNum]
                                    normDist = center_distance(rG, rD)
                                    normDist /= diag(rG) + diag(rD)
                                    normDist *= 2.0
                                    if normDist < eval_hparams['EV_PARAM_IND_CENTER_DIFF_THR']:
                                        gtRectMat[gtNum] = 1
                                        detRectMat[detNum] = 1
                                        recallAccum += eval_hparams['MTYPE_OO_O']
                                        precisionAccum += eval_hparams['MTYPE_OO_O']
                                        pairs.append({'gt': gtNum, 'det': detNum, 'type': 'OO'})
                                        evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(detNum) + "\n"
                                    else:
                                        evaluationLog += "Match Discarded GT #" + \
                                            str(gtNum) + " with Det #" + str(detNum) + \
                                            " normDist: " + str(normDist) + " \n"
                                else:
                                    evaluationLog += "Match Discarded GT #" + \
                                        str(gtNum) + " with Det #" + str(detNum) + " not single overlap\n"
                # Find one-to-many matches
                evaluationLog += "Find one-to-many matches\n"
                for gtNum in range(len(gtRects)):
                    if gtNum not in gtDontCareRectsNum:
                        match, matchesDet = one_to_many_match(gtNum)
                        if match is True:
                            evaluationLog += "num_overlaps_gt=" + str(num_overlaps_gt(gtNum))
                            # deteval에서는 일대일(one-to-one) 매핑으로 표시하기 전에 유효성 검사를 진행
                            if num_overlaps_gt(gtNum) >= 2:
                                gtRectMat[gtNum] = 1
                                recallAccum += (eval_hparams['MTYPE_OO_O'] if len(matchesDet)
                                                == 1 else eval_hparams['MTYPE_OM_O'])
                                precisionAccum += (eval_hparams['MTYPE_OO_O'] if len(matchesDet)
                                                   == 1 else eval_hparams['MTYPE_OM_O']*len(matchesDet))
                                pairs.append({'gt': gtNum, 'det': matchesDet,
                                             'type': 'OO' if len(matchesDet) == 1 else 'OM'})
                                for detNum in matchesDet:
                                    detRectMat[detNum] = 1
                                evaluationLog += "Match GT #" + str(gtNum) + " with Det #" + str(matchesDet) + "\n"
                            else:
                                evaluationLog += "Match Discarded GT #" + \
                                    str(gtNum) + " with Det #" + str(matchesDet) + " not single overlap\n"

                # Find many-to-one matches
                evaluationLog += "Find many-to-one matches\n"
                for detNum in range(len(detRects)):
                    if detNum not in detDontCareRectsNum:
                        match, matchesGt = many_to_one_match(detNum)
                        if match is True:
                          # deteval에서는 일대일(one-to-one) 매핑으로 표시하기 전에 유효성 검사를 진행
                            if num_overlaps_det(detNum) >= 2:
                                detRectMat[detNum] = 1
                                recallAccum += (eval_hparams['MTYPE_OO_O'] if len(matchesGt)
                                                == 1 else eval_hparams['MTYPE_OM_M']*len(matchesGt))
                                precisionAccum += (eval_hparams['MTYPE_OO_O']
                                                   if len(matchesGt) == 1 else eval_hparams['MTYPE_OM_M'])
                                pairs.append({'gt': matchesGt, 'det': detNum,
                                             'type': 'OO' if len(matchesGt) == 1 else 'MO'})
                                for gtNum in matchesGt:
                                    gtRectMat[gtNum] = 1
                                evaluationLog += "Match GT #" + str(matchesGt) + " with Det #" + str(detNum) + "\n"
                            else:
                                evaluationLog += "Match Discarded GT #" + \
                                    str(matchesGt) + " with Det #" + str(detNum) + " not single overlap\n"

                # 해당 샘플에 대한 최종 메트릭 계산
                numGtCare = (len(gtRects) - len(gtDontCareRectsNum))
                if numGtCare == 0:
                    recall = float(1)
                    precision = float(0) if len(detRects) > 0 else float(1)
                else:
                    recall = float(recallAccum) / numGtCare
                    precision = float(0) if (len(detRects) - len(detDontCareRectsNum)
                                             ) == 0 else float(precisionAccum) / (len(detRects) - len(detDontCareRectsNum))
                hmean = 0 if (precision + recall) == 0 else 2.0 * precision * recall / (precision + recall)

        # 전체 데이터에 대한 최종 메트릭 계산
        methodRecallSum += recallAccum
        methodPrecisionSum += precisionAccum
        numGt += len(gtRects) - len(gtDontCareRectsNum)
        numDet += len(detRects) - len(detDontCareRectsNum)

        perSampleMetrics[sample_name] = {
            'precision': precision,
            'recall': recall,
            'hmean': hmean,
            'pairs': pairs,
            'recall_matrix': [] if len(detRects) > 100 else recallMat.tolist(),
            'precision_matrix': [] if len(detRects) > 100 else precisionMat.tolist(),
            'gt_bboxes': gtPolPoints,
            'det_bboxes': detPolPoints,
            'gt_dont_care': gtDontCareRectsNum,
            'det_dont_care': detDontCareRectsNum,
        }

        if verbose:
            perSampleMetrics[sample_name].update(evaluation_log=evaluationLog)

    methodRecall = 0 if numGt == 0 else methodRecallSum/numGt
    methodPrecision = 0 if numDet == 0 else methodPrecisionSum/numDet
    methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
        methodRecall * methodPrecision / (methodRecall + methodPrecision)

    methodMetrics = {'precision': methodPrecision, 'recall': methodRecall, 'hmean': methodHmean}

    resDict = {'calculated': True, 'Message': '', 'total': methodMetrics,
               'per_sample': perSampleMetrics, 'eval_hparams': eval_hparams}

    return resDict

# %%


# 각 언어에 대해 평가 수행
for lang in languages:
    # ground truth와 prediction 파일 경로 설정
    gt_path = f"./data/{lang}_receipt/ufo/val.json"
    pred_path = f"./data/{lang}_receipt/ufo/{lang}_val_predict.json"

    # gt와 pred 데이터를 읽어오기
    gt_data = read_json(gt_path)
    pred_data = read_json(pred_path)

    # 필요한 형식으로 gt와 pred 데이터를 변환
    gt_dict = {image_id: [word['points'] for word in data['words'].values()]
               for image_id, data in gt_data['images'].items()}
    pred_dict = {image_id: [word['points'] for word in data['words'].values()]
                 for image_id, data in pred_data['images'].items()}

    # 평가 수행
    eval_params = default_evaluation_params()
    results = calc_deteval_metrics(pred_dict, gt_dict, transcriptions_dict=None,
                                   eval_hparams=eval_params, bbox_format='rect', verbose=False)

    # 언어별 결과 저장
    overall_results[lang] = {
        'precision': results['total']['precision'],
        'recall': results['total']['recall'],
        'hmean': results['total']['hmean']
    }

    # 언어별 상세 출력
    print(f"\nResults for {lang.capitalize()}:")
    print(f"Precision: {overall_results[lang]['precision']:.4f}")
    print(f"Recall: {overall_results[lang]['recall']:.4f}")
    print(f"Hmean: {overall_results[lang]['hmean']:.4f}")

# %%

# 전체 평균 계산
average_precision = np.mean([res['precision'] for res in overall_results.values()])
average_recall = np.mean([res['recall'] for res in overall_results.values()])
average_hmean = np.mean([res['hmean'] for res in overall_results.values()])

print("\nOverall Average Metrics across all languages:")
print(f"Average Precision: {average_precision:.4f}")
print(f"Average Recall: {average_recall:.4f}")
print(f"Average Hmean: {average_hmean:.4f}")
