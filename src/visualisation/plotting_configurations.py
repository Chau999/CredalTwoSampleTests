"""
linestyle reference
https://matplotlib.org/stable/gallery/lines_bars_and_markers/linestyles.html
"""

LINEWIDTH = 2.0
powers = [0.66, 0.75, 1.0]

method_linestyle = dict()
method_linestyle["MMD"] = "-"
method_linestyle["MMDq"] = ".-"
method_linestyle["MMDqstar"] = "*-"
# for power in powers:
#     method_linestyle[f"2S-SplitMMD(n=m^{power})"] = "^--"
#     method_linestyle[f"2S-DoubleDipMMD(n=m^{power})"] = "h--"

method_linestyle["2S-SplitMMD(n=m^0.5)"] = ".--"
method_linestyle[f"2S-SplitMMD(n=m^{powers[0]})"] = "^--"
method_linestyle[f"2S-SplitMMD(n=m^{powers[1]})"] = "h--"
method_linestyle[f"2S-SplitMMD(n=m^{powers[2]})"] = "D--"
method_linestyle["CMMD(0.5)"] = ".--"
method_linestyle["CMMD(0.66)"] = "^--"
method_linestyle["CMMD(0.75)"] = "h--"
method_linestyle["CMMD(1.0)"] = "D--"


method_color = dict()
method_color["MMD"] = "black"
method_color["MMDq"] = "slategrey"
method_color["MMDqstar"] = "lime"
method_color["2S-SplitMMD(n=m^0.5)"] = "black"
method_color[f"2S-SplitMMD(n=m^{powers[0]})"] = "magenta"
method_color[f"2S-SplitMMD(n=m^{powers[1]})"] = "orangered"
method_color[f"2S-SplitMMD(n=m^{powers[2]})"] = "darkred"
method_color[f"2S-DoubleDipMMD(n=m^{powers[0]})"] = "aqua"
method_color[f"2S-DoubleDipMMD(n=m^{powers[1]})"] = "royalblue"
method_color[f"2S-DoubleDipMMD(n=m^{powers[2]})"] = "darkblue"
method_color["CMMD(0.5)"] = "black"
method_color["CMMD(0.66)"] = "magenta"
method_color["CMMD(0.75)"] = "orangered"
method_color["CMMD(1.0)"] = "darkred"



method_label = dict()
method_label["MMD"] = "MMD"
method_label["MMDq"] = "MMDQ"
method_label["MMDqstar"] = "MMDQ*"
method_label["2S-SplitMMD(n=m^0.5)"] = "CMMD(0.5)"
method_label[f"2S-SplitMMD(n=m^{powers[0]})"] = "CMMD(0.33)"
method_label[f"2S-SplitMMD(n=m^{powers[1]})"] = "CMMD(0.25)"
method_label[f"2S-SplitMMD(n=m^{powers[2]})"] = "CMMD(0)"
method_label[f"2S-DoubleDipMMD(n=m^{powers[0]})"] = f"ddip({powers[0]})"
method_label[f"2S-DoubleDipMMD(n=m^{powers[1]})"] = f"ddip({powers[1]})"
method_label[f"2S-DoubleDipMMD(n=m^{powers[2]})"] = f"ddip({powers[2]})"
method_label["CMMD(0.5)"] = "CMMD(0.5)"
method_label["CMMD(0.66)"] = "CMMD(0.66)"
method_label["CMMD(0.75)"] = "CMMD(0.75)"
method_label["CMMD(1.0)"] = "CMMD(1.0)"
