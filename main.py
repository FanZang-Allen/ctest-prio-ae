# simulate prio techniques on ctest dataset
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import ordering, parsing_utils, peer, utils
from constant import *
from pinput import *
from metric import Metric

TCPS = pinput["tcps"][1]


def load_prio_data(tcp):
    print("load " + tcp + " data")
    if tcp in [RANDOMIZED]:
        return {}
    elif tcp in [T_CC_M, A_CC_M]:
        return parsing_utils.code_coverage("method")
    elif tcp in [T_CC_S, A_CC_S]:
        return parsing_utils.code_coverage("stmt")
    elif tcp in [T_PC, A_PC]:
        return parsing_utils.param_coverage(change_aware=False)
    elif tcp in [T_PCC, A_PCC]:
        return parsing_utils.param_coverage(change_aware=True)
    elif tcp in [T_ST_M, A_ST_M]:
        return parsing_utils.config_trace_coverage(change_aware=False)
    elif tcp in [T_STC_M, A_STC_M]:
        return parsing_utils.config_trace_coverage(change_aware=True)
    elif tcp == QTF:
        return parsing_utils.default_runtime()
    elif tcp == IR_TEST_LOW:
        return parsing_utils.ir_model("test", LOW)
    elif tcp == IR_TEST_HIGH:
        return parsing_utils.ir_model("test", HIGH)
    elif tcp in [T_PF_ALL, A_PF_ALL]:
        return peer.pf_all()
    elif tcp in [T_PF_DP, A_PF_DP]:
        return peer.pf_dp()
    elif tcp in [T_PF_PC, A_PF_PC]:
        return peer.pf_pc()
    elif tcp in [T_PF_RC, A_PF_RC]:
        return peer.pf_rc()
    elif tcp in [T_PF_PCC, A_PF_PCC]:
        return peer.pf_pcc()
    elif tcp in [T_PF_RCC, A_PF_RCC]:
        return peer.pf_rcc()
    elif tcp in [CB_CY]:
        return parsing_utils.cyclomatic_complexity()
    elif tcp in [CB_CY_HA_S_V, CB_CY_HA_S_L, CB_CY_HA_S_N, CB_CY_HA_S_VN, CB_CY_HA_S_VL]:
        cb_data = parsing_utils.cyclomatic_complexity()
        metric = tcp.split("_")[4:]
        ha_data = parsing_utils.halstead_metric("simple", metric)
        for key, value in ha_data.items():
            ha_data[key] = cb_data[key] + value
        return ha_data
    elif tcp in [CB_HA_S]:
        return parsing_utils.halstead_metric("simple")
    elif tcp in [CB_HA_S_D, CB_HA_S_E, CB_HA_S_L, CB_HA_S_N, CB_HA_S_V]:
        metric = tcp.split("_")[3:]
        return parsing_utils.halstead_metric("simple", metric)
    elif tcp in [CB_HA_I]:
        return parsing_utils.halstead_metric("inherit")
    elif tcp in [CB_HA_I_D, CB_HA_I_E, CB_HA_I_L, CB_HA_I_N, CB_HA_I_V]:
        metric = tcp.split("_")[3:]
        return parsing_utils.halstead_metric("inherit", metric)
    elif tcp in [CB_CY_HA_I_V, CB_CY_HA_I_L, CB_CY_HA_I_N, CB_CY_HA_I_VN, CB_CY_HA_I_VL]:
        cb_data = parsing_utils.cyclomatic_complexity()
        metric = tcp.split("_")[4:]
        ha_data = parsing_utils.halstead_metric("inherit", metric)
        for key, value in ha_data.items():
            ha_data[key] = cb_data[key] + value
        return ha_data

    # hybrid
    data = {"d1": {}, "d2": {}, "d3": {}}
    if tcp in [T_CC_M_DIV, A_CC_M_DIV, T_CC_M_BT, A_CC_M_BT]:
        data["d1"] = parsing_utils.code_coverage("method")
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [T_CC_S_DIV, A_CC_S_DIV, T_CC_S_BT, A_CC_S_BT]:
        data["d1"] = parsing_utils.code_coverage("stmt")
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [T_PC_DIV, A_PC_DIV, T_PC_BT, A_PC_BT]:
        data["d1"] = parsing_utils.param_coverage(change_aware=False)
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [T_PCC_DIV, A_PCC_DIV, T_PCC_BT, A_PCC_BT]:
        data["d1"] = parsing_utils.param_coverage(change_aware=True)
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [T_ST_M_DIV, A_ST_M_DIV, T_ST_M_BT, A_ST_M_BT]:
        data["d1"] = parsing_utils.config_trace_coverage(change_aware=False)
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [T_STC_M_DIV, A_STC_M_DIV, T_STC_M_BT, A_STC_M_BT]:
        data["d1"] = parsing_utils.config_trace_coverage(change_aware=True)
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [IR_TEST_LOW_DIV, IR_TEST_LOW_BT]:
        data["d1"] = parsing_utils.ir_model("test", LOW)
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [IR_TEST_HIGH_DIV, IR_TEST_HIGH_BT]:
        data["d1"] = parsing_utils.ir_model("test", HIGH)
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [RANDOMIZED_DIV, QTF_DIV, RANDOMIZED_BT, QTF_BT]:
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [T_PF_ALL_DIV, A_PF_ALL_DIV, T_PF_ALL_BT, A_PF_ALL_BT]:
        data["d1"] = peer.pf_all()
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [T_PF_DP_DIV, A_PF_DP_DIV, T_PF_DP_BT, A_PF_DP_BT]:
        data["d1"] = peer.pf_dp()
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [T_PF_PC_DIV, A_PF_PC_DIV, T_PF_PC_BT, A_PF_PC_BT]:
        data["d1"] = peer.pf_pc()
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [T_PF_RC_DIV, A_PF_RC_DIV, T_PF_RC_BT, A_PF_RC_BT]:
        data["d1"] = peer.pf_rc()
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [T_PF_PCC_DIV, A_PF_PCC_DIV, T_PF_PCC_BT, A_PF_PCC_BT]:
        data["d1"] = peer.pf_pcc()
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [T_PF_RCC_DIV, A_PF_RCC_DIV, T_PF_RCC_BT, A_PF_RCC_BT]:
        data["d1"] = peer.pf_rcc()
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [CB_CY_DIV, CB_CY_BT]:
        data["d1"] = parsing_utils.cyclomatic_complexity()
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [CB_HA_S_DIV, CB_HA_S_BT]:
        data["d1"] = parsing_utils.halstead_metric("simple")
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [CB_HA_I_DIV, CB_HA_I_BT]:
        data["d1"] = parsing_utils.halstead_metric("inherit")
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [CB_HA_S_D_DIV, CB_HA_S_E_DIV, CB_HA_S_L_DIV, CB_HA_S_N_DIV, CB_HA_S_V_DIV, CB_HA_S_D_BT, CB_HA_S_E_BT, CB_HA_S_L_BT, CB_HA_S_N_BT, CB_HA_S_V_BT]:
        metric = tcp.split("_")[3:-1]
        data["d1"] = parsing_utils.halstead_metric("simple", metric)
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [CB_CY_HA_S_V_DIV, CB_CY_HA_S_L_DIV, CB_CY_HA_S_N_DIV, CB_CY_HA_S_VN_DIV, CB_CY_HA_S_VL_DIV, CB_CY_HA_S_V_BT, CB_CY_HA_S_L_BT, CB_CY_HA_S_N_BT, CB_CY_HA_S_VN_BT, CB_CY_HA_S_VL_BT]:
        metric = tcp.split("_")[4:-1]
        cb_data = parsing_utils.cyclomatic_complexity()
        ha_data = parsing_utils.halstead_metric("simple", metric)
        for key, value in ha_data.items():
            ha_data[key] = cb_data[key] + value
        data["d1"] = ha_data
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [CB_HA_I_D_DIV, CB_HA_I_E_DIV, CB_HA_I_L_DIV, CB_HA_I_N_DIV, CB_HA_I_V_DIV, CB_HA_I_D_BT, CB_HA_I_E_BT, CB_HA_I_L_BT, CB_HA_I_N_BT, CB_HA_I_V_BT]:
        metric = tcp.split("_")[3:-1]
        data["d1"] = parsing_utils.halstead_metric("inherit", metric)
        data["d2"] = parsing_utils.default_runtime()
    elif tcp in [CB_CY_HA_I_V_DIV, CB_CY_HA_I_L_DIV, CB_CY_HA_I_N_DIV, CB_CY_HA_I_VN_DIV, CB_CY_HA_I_VL_DIV, CB_CY_HA_I_V_BT, CB_CY_HA_I_L_BT, CB_CY_HA_I_N_BT, CB_CY_HA_I_VN_BT, CB_CY_HA_I_VL_BT]:
        metric = tcp.split("_")[4:-1]
        cb_data = parsing_utils.cyclomatic_complexity()
        ha_data = parsing_utils.halstead_metric("inherit", metric)
        for key, value in ha_data.items():
            ha_data[key] = cb_data[key] + value
        data["d1"] = ha_data
        data["d2"] = parsing_utils.default_runtime()
    else:
        exit("load_prio_data: unknown tcp: {}".format(tcp))
    return data


def prioritize(imgname, tcp, testinfo):
    # given a conf change and a tcp, return result logs
    prio_data = load_prio_data(tcp)
    prio = ordering.Prio(tcp, imgname, testinfo, prio_data)
    if not tcp.endswith("_div") and not tcp.endswith("_bt"):
        # basic
        if tcp.startswith("add"):
            prio.additional()
        elif tcp.startswith("tot"):
            prio.total()
        elif tcp.startswith("randomized"):
            prio.randomized()
        elif tcp.startswith("qtf"):
            prio.qtf()
        elif tcp.startswith("ir"):
            prio.ir()
        elif tcp.startswith("cb"):
            prio.complexity_based()
        else:
            exit("prioritize: unknown basic tcp: {}".format(tcp))
    else:
        # hybrid
        if tcp.startswith("add"):
            prio.additional_hybrid()
        elif tcp.startswith("tot"):
            prio.total_hybrid()
        elif tcp in [IR_TEST_LOW_DIV, IR_TEST_HIGH_DIV, IR_TEST_LOW_BT, IR_TEST_HIGH_BT]:
            prio.ir_hybrid()
        elif tcp in [RANDOMIZED_DIV, RANDOMIZED_BT]:
            prio.randomized_hybrid()
        elif tcp in [QTF_DIV, QTF_BT]:
            prio.qtf_hybrid()
        elif tcp.startswith("cb"):
            prio.complexity_hybrid()
        else:
            exit("prioritize: unknown hybrid tcp: {}".format(tcp))
    return prio.logs


def main():
    s = time.time()
    dataset = parsing_utils.parse_docker_dataset()
    executor = ThreadPoolExecutor(max_workers=10)
    all_logs = [executor.submit(run_tcp, tcp, dataset).result() for tcp in TCPS]

    # log results
    assert len(all_logs) == len(TCPS), "number of run TCP not match."
    for per_tcp_logs in all_logs:
        assert pinput["nrun"] * len(dataset) == len(per_tcp_logs), "number of config change * seed not match."
        for line in per_tcp_logs:
            logging.info(line)
    logging.info("total-time: {}".format(time.time()-s))


def run_tcp(tcp, dataset):
    per_tcp_logs = []
    data_for_tcp = load_prio_data(tcp)
    for img, fdata in dataset.items():
        per_tcp_logs += prioritize(img, tcp, fdata["testinfo"])
    return per_tcp_logs

if __name__ == '__main__':
    main()