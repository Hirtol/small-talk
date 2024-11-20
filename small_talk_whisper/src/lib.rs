//! Small wrapper to dynamically link to whisper-cpp, as it conflicts with the GGML compiled for llama.cpp

use std::ffi::{c_char, CStr, CString};
pub use whisper_rs::*;

#[repr(C)]
pub struct WhisperCtx {
    ctx: WhisperContext,
    state: WhisperState,
    params: FullParams<'static, 'static>
}

#[no_mangle]
pub unsafe extern "C" fn create_whisper(path: *const c_char) -> *mut WhisperCtx {
    let path = CStr::from_ptr(path);
    
    let ctx = WhisperContext::new_with_params(&path.to_string_lossy(), WhisperContextParameters::default()).unwrap();
    let mut state = ctx.create_state().unwrap();
    let mut params = FullParams::new(SamplingStrategy::Greedy {best_of: 1});
    params.set_language(Some("en"));

    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    let whisper = Box::new(WhisperCtx {
        ctx,
        state,
        params,
    });
    Box::leak(whisper)
}

#[no_mangle]
pub unsafe extern "C" fn parse_tokens(whisper: &mut WhisperCtx, samples: *const i16, sample_cnt: usize) -> CString {
    let data = std::slice::from_raw_parts(samples, sample_cnt);
    let mut inter_samples = vec![Default::default(); data.len()];

    whisper_rs::convert_integer_to_float_audio(&data, &mut inter_samples)
        .expect("failed to convert audio data");
    let samples = whisper_rs::convert_stereo_to_mono_audio(&inter_samples).unwrap_or(inter_samples);

    // now we can run the model
    // note the key we use here is the one we created above
    whisper.state
        .full(whisper.params.clone(), &samples[..])
        .expect("failed to run model");

    // fetch the results
    let num_segments = whisper.state
        .full_n_segments()
        .expect("failed to get number of segments");
    let mut full_text = String::new();
    for i in 0..num_segments {
        let segment =  whisper.state
            .full_get_segment_text(i)
            .expect("failed to get segment");
        full_text += &segment;
        let start_timestamp =  whisper.state
            .full_get_segment_t0(i)
            .expect("failed to get segment start timestamp");
        let end_timestamp =  whisper.state
            .full_get_segment_t1(i)
            .expect("failed to get segment end timestamp");
        println!("[{} - {}]: {}", start_timestamp, end_timestamp, segment);
    }
    
    CString::new(full_text).expect("Fail")
}


#[no_mangle]
pub unsafe extern "C" fn free_whisper(whisper: *mut WhisperCtx) {
    Box::from_raw(whisper);
}