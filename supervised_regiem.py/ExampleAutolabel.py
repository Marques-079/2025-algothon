from pre_proc_labelling import get_data


#Works well on Range < 12 
#Work well on Range > 12 
#Work well on Range > 25

labels = get_data(2, 9, 750, False)  #(Range setting, Instrument, endpoint, plot graph)
print(labels)