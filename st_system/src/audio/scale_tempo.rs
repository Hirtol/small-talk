//! Converted from VLC's `scaletempo.c`
//! https://code.videolan.org/vikram-kangotra/vlc/-/blob/rust-for-vlc/modules/audio_filter/scaletempo.c

pub struct Scaletempo {
    scale: f64,
    sample_rate: u32,
    channels: usize,
    ms_stride: u32,
    percent_overlap: f64,
    ms_search: u32,
    stride_frames: usize,
    overlap_frames: usize,
    search_frames: usize,
    queue: Vec<f32>,
    overlap: Vec<f32>,
    blend_table: Vec<f32>,
    window_table: Vec<f32>,
    buf_pre_corr: Vec<f32>,
    queued_frames: usize,
    to_slide_frames: usize,
    stride_error: f64,
}

impl Scaletempo {
    pub fn new(sample_rate: u32, channels: usize, ms_stride: u32, percent_overlap: f64, ms_search: u32) -> Self {
        let stride_frames = (ms_stride as f64 * sample_rate as f64 / 1000.0).round() as usize;
        let overlap_frames = (stride_frames as f64 * percent_overlap).round() as usize;
        let search_frames = (ms_search as f64 * sample_rate as f64 / 1000.0).round() as usize;

        let queue_capacity = (search_frames + stride_frames + overlap_frames) * channels;
        let mut queue = Vec::with_capacity(queue_capacity);
        queue.resize(queue_capacity, 0.0);

        let mut blend_table = Vec::new();
        if overlap_frames > 0 {
            blend_table.reserve(overlap_frames * channels);
            for i in 0..overlap_frames {
                let factor = i as f32 / overlap_frames as f32;
                for _ in 0..channels {
                    blend_table.push(factor);
                }
            }
        }

        let mut window_table = Vec::new();
        if overlap_frames > 1 {
            window_table.reserve((overlap_frames - 1) * channels);
            for i in 1..overlap_frames {
                let v = i as f32 * (overlap_frames - i) as f32;
                for _ in 0..channels {
                    window_table.push(v);
                }
            }
        }

        let buf_pre_corr = vec![0.0; window_table.len()];
        let overlap = vec![0.0; overlap_frames * channels];

        Scaletempo {
            scale: 1.0,
            sample_rate,
            channels,
            ms_stride,
            percent_overlap,
            ms_search,
            stride_frames,
            overlap_frames,
            search_frames,
            queue,
            overlap,
            blend_table,
            window_table,
            buf_pre_corr,
            queued_frames: 0,
            to_slide_frames: 0,
            stride_error: 0.0,
        }
    }

    fn standing_frames(&self) -> usize {
        self.stride_frames - self.overlap_frames
    }

    fn required_frames(&self) -> usize {
        self.search_frames + self.stride_frames + self.overlap_frames
    }

    fn queue_capacity(&self) -> usize {
        self.queue.len() / self.channels
    }

    fn update_buf_pre_corr(&mut self) {
        if self.overlap_frames > 1 {
            for i in 0..self.buf_pre_corr.len() {
                let j = i + self.channels;
                self.buf_pre_corr[i] = self.window_table[i] * self.overlap[j];
            }
        }
    }

    fn best_overlap_offset(&self) -> usize {
        if self.overlap_frames <= 1 || self.search_frames == 0 {
            return 0;
        }

        let mut best_corr = f32::MIN;
        let mut best_off = 0;

        for off in 0..self.search_frames {
            let mut corr = 0.0;
            let start = off * self.channels + self.channels;
            for i in 0..self.buf_pre_corr.len() {
                corr += self.buf_pre_corr[i] * self.queue[start + i];
            }
            if corr > best_corr {
                best_corr = corr;
                best_off = off;
            }
        }

        best_off
    }

    fn fill_queue(&mut self, input: &[f32], offset: usize) -> usize {
        let mut consumed = 0;
        let samples_offset = offset;

        if self.to_slide_frames > 0 {
            if self.to_slide_frames < self.queued_frames {
                let samples_to_remove = self.to_slide_frames * self.channels;
                let samples_remaining = self.queued_frames * self.channels - samples_to_remove;
                for i in 0..samples_remaining {
                    self.queue[i] = self.queue[i + samples_to_remove];
                }
                self.queued_frames -= self.to_slide_frames;
                self.to_slide_frames = 0;
            } else {
                let frames_remove = self.queued_frames;
                self.to_slide_frames -= frames_remove;
                self.queued_frames = 0;
                let available = input.len().saturating_sub(samples_offset + consumed);
                let available_frames = available / self.channels;
                let skip_frames = self.to_slide_frames.min(available_frames);
                let skip_samples = skip_frames * self.channels;
                consumed += skip_samples;
                self.to_slide_frames -= skip_frames;
            }
        }

        let available = input.len().saturating_sub(samples_offset + consumed);
        if available > 0 {
            let available_frames = available / self.channels;
            let free_frames = self.queue_capacity() - self.queued_frames;
            let frames_to_append = available_frames.min(free_frames);
            if frames_to_append > 0 {
                let samples_to_append = frames_to_append * self.channels;
                let start = self.queued_frames * self.channels;
                let end = start + samples_to_append;
                let src_start = samples_offset + consumed;
                let src_end = src_start + samples_to_append;
                self.queue[start..end].copy_from_slice(&input[src_start..src_end]);
                self.queued_frames += frames_to_append;
                consumed += samples_to_append;
            }
        }

        consumed
    }

    pub fn process(&mut self, input: &[f32], speed: f64) -> Vec<f32> {
        let mut output = Vec::new();
        let mut offset = 0;

        offset += self.fill_queue(input, offset);

        while self.queued_frames >= self.required_frames() {
            self.scale = speed;

            let best_offset_frames = if self.overlap_frames > 0 && self.search_frames > 0 {
                self.best_overlap_offset()
            } else {
                0
            };

            if self.overlap_frames > 0 {
                let start_index = best_offset_frames * self.channels;
                for i in 0..self.overlap_frames * self.channels {
                    let blend_factor = self.blend_table[i];
                    let from_overlap = self.overlap[i];
                    let from_queue = self.queue[start_index + i];
                    let out_sample = from_overlap * (1.0 - blend_factor) + from_queue * blend_factor;
                    output.push(out_sample);
                }
            }

            let standing_frames = self.standing_frames();
            if standing_frames > 0 {
                let start_index = (best_offset_frames + self.overlap_frames) * self.channels;
                let end_index = start_index + standing_frames * self.channels;
                output.extend_from_slice(&self.queue[start_index..end_index]);
            }

            if self.overlap_frames > 0 {
                let start_index = (best_offset_frames + self.stride_frames) * self.channels;
                let end_index = start_index + self.overlap_frames * self.channels;
                self.overlap.copy_from_slice(&self.queue[start_index..end_index]);
                self.update_buf_pre_corr();
            }

            let frames_to_consume = self.stride_frames as f64 * self.scale + self.stride_error;
            let whole_frames = frames_to_consume as usize;
            self.stride_error = frames_to_consume - whole_frames as f64;
            self.to_slide_frames = whole_frames;

            let consumed = self.fill_queue(input, offset);
            offset += consumed;
        }

        let consumed = self.fill_queue(input, offset);
        offset += consumed;

        output
    }
}