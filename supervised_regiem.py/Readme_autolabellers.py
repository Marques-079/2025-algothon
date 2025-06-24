from pre_proc_labelling import get_data
from pre_proc_labelling_long import plot_all_regimes_long
from pre_proc_labelling_short import plot_all_regimes_short


#Works well on Range < 12 
#Work well on Range > 12 
#Work well on Range > 25

#V1
#labels = get_data(1, 10, 600, True)  #(Range setting, Instrument, endpoint, plot graph)
#print(labels)

#V3
'''
If you want a specific instrument parse the instance at the end of the function call eg. labels = plot_all_regimes_long(750, True, 42) where 42 is the instance. 
Note it is Zero indexed, so first instrument is 0

short -> Plots all uptrends can be used for a HFT model rather than regiems based
long -> regiems based model, will hold trends over long periods of time expect a range of 1 - 7 trends over 750 days timeframe

def example_entry(end_point: int, plot_graph: bool = True, inst: Optional[int] = None) -> None
'''
#labels = plot_all_regimes_short(750, True, 0)

labels = plot_all_regimes_long(500, False, 1)
print(len(labels))