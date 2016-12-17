local div_utils = {}

function div_utils.cumulative_div(beam_seq_table,t,divm,divmode)
--INPUTS:
--beam_seq_table: table containing the beams of M-diverse groups
--divm          : index of the group for which diversity needs to be computed
--t             : time slice that needs to be considered. 
--OUTPUTS:
--cumulative_diversity: diversity between the current beam and all other previous beam (a value)
  if divmode == 0 then
    return 1
  end
  local local_time = t-divm+1
  if local_time==1 then 
    return 1 
  else
    current_beam = beam_seq_table[divm][{{1,local_time-1},}]
    temp_div = 0
    for i=1,divm-1 do
      temp_div = temp_div + torch.ne(current_beam,beam_seq_table[i][{{1,local_time-1},}]):sum()/current_beam:nElement()
    end
    return torch.exp(-temp_div/(divm-1))
  end
end

return div_utils
