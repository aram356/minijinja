use std::ptr::addr_of_mut;
use std::{fmt, io};

use crate::error::{Error, ErrorKind};
use crate::utils::AutoEscape;
use crate::value::Value;

/// How should output be captured?
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[cfg_attr(feature = "unstable_machinery_serde", derive(serde::Serialize))]
pub enum CaptureMode {
    Capture,
    #[allow(unused)]
    Discard,
}

/// An abstraction over [`fmt::Write`](std::fmt::Write) for the rendering.
///
/// This is a utility type used in the engine which can be written into like one
/// can write into an [`std::fmt::Write`] value.  It's primarily used internally
/// in the engine but it's also passed to the custom formatter function.
pub struct Output<'a> {
    w: *mut (dyn fmt::Write + 'a),
    target: *mut (dyn fmt::Write + 'a),
    capture_stack: Vec<Option<String>>,
    /// Bytes written to the base target so far, tracked only when that target
    /// is itself an in-memory buffer (see [`Output::new_buffered`]).
    base_written: Option<usize>,
}

impl<'a> Output<'a> {
    /// Creates a new output.
    pub(crate) fn new(w: &'a mut (dyn fmt::Write + 'a)) -> Self {
        Self {
            w,
            target: w,
            capture_stack: Vec::new(),
            base_written: None,
        }
    }

    /// Creates a new output whose base target is an in-memory buffer.
    ///
    /// Rendering into a buffer -- a macro body, a block rendered to a string --
    /// never reaches the caller's writer, so a bounded writer supplied by the
    /// caller cannot see it grow.  Marking the base as buffered exposes the
    /// accumulated size through [`buffered_len`](Self::buffered_len) so the
    /// evaluator can bound it while it is being built.
    // Only macro bodies render into a buffer today, so this is unused when
    // macros are compiled out.
    #[allow(dead_code)]
    pub(crate) fn new_buffered(w: &'a mut (dyn fmt::Write + 'a)) -> Self {
        Self {
            w,
            target: w,
            capture_stack: Vec::new(),
            base_written: Some(0),
        }
    }

    /// Creates a null output that writes nowhere.
    pub(crate) fn null() -> Self {
        // The null writer also has a single entry on the discarding capture
        // stack.  In fact, `w` is more or less useless here as we always
        // shadow it.  This is done so that `is_discarding` returns true.
        Self {
            w: NullWriter::get_mut(),
            target: NullWriter::get_mut(),
            capture_stack: vec![None],
            base_written: None,
        }
    }

    /// Returns how many bytes the current write target holds, if that target is
    /// an in-memory buffer.
    ///
    /// `None` means output is streaming to the caller's writer, or is being
    /// discarded; in both cases there is no buffer here to bound.
    pub(crate) fn buffered_len(&self) -> Option<usize> {
        match self.capture_stack.last() {
            Some(Some(buf)) => Some(buf.len()),
            Some(None) => None,
            None => self.base_written,
        }
    }

    /// Begins capturing into a string or discard.
    pub(crate) fn begin_capture(&mut self, mode: CaptureMode) {
        self.capture_stack.push(match mode {
            CaptureMode::Capture => Some(String::new()),
            CaptureMode::Discard => None,
        });
        self.retarget();
    }

    /// Ends capturing and returns the captured string as value.
    pub(crate) fn end_capture(&mut self, auto_escape: AutoEscape) -> Value {
        let rv = if let Some(captured) = self.capture_stack.pop().unwrap() {
            if !matches!(auto_escape, AutoEscape::None) {
                Value::from_safe_string(captured)
            } else {
                Value::from(captured)
            }
        } else {
            Value::UNDEFINED
        };
        self.retarget();
        rv
    }

    fn retarget(&mut self) {
        self.target = match self.capture_stack.last_mut() {
            Some(Some(stream)) => stream,
            Some(None) => NullWriter::get_mut(),
            None => self.w,
        };
    }

    #[inline(always)]
    fn target(&mut self) -> &mut dyn fmt::Write {
        // SAFETY: this is safe because we carefully maintain the capture stack
        // to update self.target whenever it's modified
        unsafe { &mut *self.target }
    }

    /// Returns `true` if the output is discarding.
    #[inline(always)]
    #[allow(unused)]
    pub(crate) fn is_discarding(&self) -> bool {
        matches!(self.capture_stack.last(), Some(None))
    }

    /// Records bytes written straight to a buffered base target.
    ///
    /// Writes into a capture frame need no bookkeeping: the frame's `String` is
    /// its own counter.  Only the base target is opaque, so its size is tracked
    /// here.
    #[inline(always)]
    fn note_base_write(&mut self, len: usize) {
        if self.capture_stack.is_empty() {
            if let Some(ref mut written) = self.base_written {
                *written = written.saturating_add(len);
            }
        }
    }

    /// Writes some data to the underlying buffer contained within this output.
    #[inline]
    pub fn write_str(&mut self, s: &str) -> fmt::Result {
        self.note_base_write(s.len());
        self.target().write_str(s)
    }

    /// Writes some formatted information into this instance.
    #[inline]
    pub fn write_fmt(&mut self, a: fmt::Arguments<'_>) -> fmt::Result {
        fmt::Write::write_fmt(self, a)
    }
}

impl fmt::Write for Output<'_> {
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.note_base_write(s.len());
        fmt::Write::write_str(self.target(), s)
    }

    #[inline]
    fn write_char(&mut self, c: char) -> fmt::Result {
        self.note_base_write(c.len_utf8());
        fmt::Write::write_char(self.target(), c)
    }
}

pub struct NullWriter;

impl NullWriter {
    /// Returns a reference to the null writer.
    pub fn get_mut() -> &'static mut NullWriter {
        static mut NULL_WRITER: NullWriter = NullWriter;
        // SAFETY: this is safe as the null writer is a ZST
        unsafe { &mut *addr_of_mut!(NULL_WRITER) }
    }
}

impl fmt::Write for NullWriter {
    #[inline]
    fn write_str(&mut self, _s: &str) -> fmt::Result {
        Ok(())
    }

    #[inline]
    fn write_char(&mut self, _c: char) -> fmt::Result {
        Ok(())
    }
}

pub struct WriteWrapper<W> {
    pub w: W,
    pub err: Option<io::Error>,
}

impl<W> WriteWrapper<W> {
    /// Replaces the given error with the held error if available.
    pub fn take_err(&mut self, original: Error) -> Error {
        self.err
            .take()
            .map(|io_err| {
                Error::new(ErrorKind::WriteFailure, "I/O error during rendering")
                    .with_source(io_err)
            })
            .unwrap_or(original)
    }
}

impl<W: io::Write> fmt::Write for WriteWrapper<W> {
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.w.write_all(s.as_bytes()).map_err(|e| {
            self.err = Some(e);
            fmt::Error
        })
    }

    #[inline]
    fn write_char(&mut self, c: char) -> fmt::Result {
        self.w
            .write_all(c.encode_utf8(&mut [0; 4]).as_bytes())
            .map_err(|e| {
                self.err = Some(e);
                fmt::Error
            })
    }
}
