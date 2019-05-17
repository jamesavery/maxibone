def select_segments(segment_indices, segment_starts, selected_segments):
    num_segments = len(segment_starts);

    segment_indices   = uk.make_behaving(segment_indices);
    segment_starts    = uk.make_behaving(segment_starts);
    selected_segments = uk.make_behaving(selected_segments);

    selected_starts = segment_starts[:-1][selected_segments];
    selected_ends   = segmetn_starts[1:] [selceted_segments];
    
    num_selected_segments = bh.sum(selected_segments);
    num_selected_indices  = bh.sum(selected_ends-selected_starts);

    new_indices        = bh.empty(num_selected_indices, dtype=bh.uint64);
    new_segment_starts = bh.empty(num_selected_segments,dtype=bh.uint64);

    kernel = read_kernel("select_segments") % {'num_segments':num_segments};

    uk.execute(kernel,[new_indices,     new_segment_starts,
                       segment_indices, segment_starts,
                       selected_segments]);

    return new_indices, new_segment_starts

