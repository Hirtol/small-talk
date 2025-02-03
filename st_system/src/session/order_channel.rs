use std::sync::Arc;
use tokio::sync::Mutex;
use std::collections::VecDeque;
use tokio::sync::mpsc::error::TrySendError;

/// Create a new ordered channel, where all elements can be re-arranged even after having been dispatched.
///
/// Internally it is backed by an unbounded [VecDeque].
pub fn ordered_channel<T>() -> (OrderedSender<T>, OrderedReceiver<T>) {
    let queue = Arc::new(Mutex::new(VecDeque::new()));
    // We use a channel to piggyback off their Drop handling, automatically closing both channels with an `Err`.
    let (send, recv) = tokio::sync::mpsc::channel(1);

    (OrderedSender {
        queue: queue.clone(),
        notify: send,
    }, OrderedReceiver {
        queue,
        notify: recv,
    })
}

pub struct OrderedReceiver<T> {
    queue: Arc<Mutex<VecDeque<T>>>,
    notify: tokio::sync::mpsc::Receiver<()>,
}

#[derive(Clone)]
pub struct OrderedSender<T> {
    queue: Arc<Mutex<VecDeque<T>>>,
    notify: tokio::sync::mpsc::Sender<()>,
}

impl<T> OrderedSender<T> {
    pub async fn change_queue<O>(&self, closure: impl for<'a> FnOnce(&'a mut VecDeque<T>) -> O) -> eyre::Result<O> {
        let mut q = self.queue.lock().await;
        let out = closure(&mut *q);
        // Notify the queue worker that we have added new items
        match self.notify.try_send(()) {
            Err(TrySendError::Closed(_)) => Err(eyre::eyre!("Channel was closed")),
            _ => Ok(out)
        }
    }

    pub fn is_closed(&self) -> bool {
        self.notify.is_closed()
    }
}

impl<T> OrderedReceiver<T> {
    /// Receive from the underlying queue, or `await` until a value is available.
    pub async fn recv(&mut self) -> Option<T> {
        loop {
            {
                let mut q = self.queue.lock().await;
                if let Some(value) = q.pop_front() {
                    return Some(value);
                }
            }
            // Wait for notification outside the lock to avoid deadlocks
            self.notify.recv().await;
        }
    }

    /// Clone the internal contents and return
    pub(crate) async fn modify_contents<O>(&self, func: impl FnOnce(&mut VecDeque<T>) -> O) -> O {
        let mut q = self.queue.lock().await;
        func(&mut q)
    }

    /// Returns the number of items in the queue.
    pub async fn len(&self) -> usize {
        self.queue.lock().await.len()
    }
}