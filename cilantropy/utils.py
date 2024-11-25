def get_uVPerBit(meta, metaFullPath, probe_type) -> float:
    # Returns uVPerBit conversion factor for channel 0
    # If all channels have the same gain (usually set that way for
    # 3A and NP1 probes; always true for NP2 probes), can use
    # this value for all channels.

    # first check if metadata includes the imChan0apGain key
    if "uVPerBit" in meta:
        return float(meta["uVPerBit"])

    if "imChan0apGain" in meta:
        APgain = float(meta["imChan0apGain"])
        voltage_range = float(meta["imAiRangeMax"]) - float(meta["imAiRangeMin"])
        maxInt = float(meta["imMaxInt"])
        uVPerBit = (1e6) * (voltage_range / APgain) / (2 * maxInt)

    else:
        imroList = meta["imroTbl"].split(sep=")")
        # One entry for each channel plus header entry,
        # plus a final empty entry following the last ')'
        # channel zero is the 2nd element in the list

        if probe_type == "NP21" or probe_type == "NP24":
            # NP 2.0; APGain = 80 for all channels
            # voltage range = 1V
            # 14 bit ADC
            uVPerBit = (1e6) * (1.0 / 80) / pow(2, 14)
        elif probe_type == "NP1110":
            # UHD2 with switches, special imro table with gain in header
            currList = imroList[0].split(sep=",")
            APgain = float(currList[3])
            uVPerBit = (1e6) * (1.2 / APgain) / pow(2, 10)
        else:
            # 3A, 3B1, 3B2 (NP 1.0), or other NP 1.0-like probes
            # voltage range = 1.2V
            # 10 bit ADC
            currList = imroList[1].split(
                sep=" "
            )  # 2nd element in list, skipping header
            APgain = float(currList[3])
            uVPerBit = (1e6) * (1.2 / APgain) / pow(2, 10)

    # save this value in meta
    with open(metaFullPath, "ab") as f:
        f.write(f"uVPerBit={uVPerBit}\n".encode("utf-8"))

    return uVPerBit
