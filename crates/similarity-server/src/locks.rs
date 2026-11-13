//! One lock for each record that a request is at, and none for any other (docs/decisions.md, D10 and D41).
//!
//! Within a process, requests for one record take turns: evaluating is read, compare, write, and two revisions
//! of a record must not both be compared with the same baseline. Requests for different records have nothing
//! to wait for in each other, and here they never do: a record's lock is its own, made when the first request
//! for the record arrives and forgotten when the last one has left. So the table is as large as the number of
//! requests in flight, however many records there are.
//!
//! It used to be 256 locks shared out by a hash of the id. With 64 requests in flight about one in five then
//! shared a lock with a stranger and waited out the stranger's whole request.
//!
//! The lock itself is tokio's `Mutex`, which serves those waiting in the order they came. What is here is only
//! the table: which records have one.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use tokio::sync::OwnedMutexGuard;

/// The locks of the records that requests are at right now.
#[derive(Debug, Default)]
pub struct Locks {
  table: Arc<Table>,
}

/// A record's lock, held. The next request for the record goes on when this is dropped.
#[derive(Debug)]
pub struct Hold {
  // In this order: the lock is let go before the place is given up, so that a record is never forgotten while
  // it is held.
  _lock: OwnedMutexGuard<()>,
  _place: Place,
}

impl Locks {
  /// Waits for `id`'s lock: for every request for that record that came before this one, and for nothing else.
  ///
  /// Dropped while it waits, it leaves nothing behind, and those behind it are not held up.
  pub async fn hold(&self, id: &str) -> Hold {
    let (place, lock) = self.table.join(id);
    let lock = lock.lock_owned().await;
    Hold {
      _lock: lock,
      _place: place,
    }
  }

  /// How many records have a lock: those somebody holds or waits for.
  #[cfg(test)]
  pub(crate) fn in_use(&self) -> usize {
    self.table.records().len()
  }
}

/// A record that requests are at: its lock, and how many of them there are, the holder and those waiting.
#[derive(Debug)]
struct Entry {
  /// The table's key, to be shared with each request's place.
  id: Arc<str>,
  lock: Arc<tokio::sync::Mutex<()>>,
  places: usize,
}

#[derive(Debug, Default)]
struct Table(Mutex<HashMap<Arc<str>, Entry>>);

impl Table {
  /// Nothing that holds this can panic, and a map that a panic interrupted is still a map: a poisoned lock is
  /// taken anyway, so that one failed request cannot stop every later one.
  fn records(&self) -> std::sync::MutexGuard<'_, HashMap<Arc<str>, Entry>> {
    self.0.lock().unwrap_or_else(PoisonError::into_inner)
  }

  /// Takes a place at `id`'s lock, making the lock if nobody is there.
  fn join(self: &Arc<Self>, id: &str) -> (Place, Arc<tokio::sync::Mutex<()>>) {
    let mut records = self.records();
    let (id, lock) = if let Some(entry) = records.get_mut(id) {
      entry.places = entry.places.saturating_add(1);
      (Arc::clone(&entry.id), Arc::clone(&entry.lock))
    } else {
      let entry = Entry {
        id: Arc::from(id),
        lock: Arc::default(),
        places: 1,
      };
      let made = (Arc::clone(&entry.id), Arc::clone(&entry.lock));
      records.insert(Arc::clone(&entry.id), entry);
      made
    };
    let place = Place {
      table: Arc::clone(self),
      id,
    };
    (place, lock)
  }
}

/// A request's place at a record's lock, from when it asks for the lock until it is done with it or gives up
/// waiting. The last one to go takes the record out of the table.
#[derive(Debug)]
struct Place {
  table: Arc<Table>,
  id: Arc<str>,
}

impl Drop for Place {
  fn drop(&mut self) {
    let mut records = self.table.records();
    let Some(entry) = records.get_mut(&*self.id) else {
      return;
    };
    entry.places = entry.places.saturating_sub(1);
    if entry.places == 0 {
      records.remove(&*self.id);
    }
  }
}

#[cfg(test)]
pub(crate) mod testing {
  use std::pin::Pin;
  use std::task::{Context, Poll, Waker};

  use super::{Hold, Locks};

  /// A request at a record's lock: asking for it, and once it has been given it, holding it.
  pub(crate) struct Asking<'a> {
    waiting: Option<Pin<Box<dyn Future<Output = Hold> + 'a>>>,
    holding: Option<Hold>,
  }

  impl<'a> Asking<'a> {
    /// Asks for `id`'s lock, and takes it if it is free.
    pub(crate) fn new(locks: &'a Locks, id: &'a str) -> Self {
      let mut asking = Self {
        waiting: Some(Box::pin(locks.hold(id))),
        holding: None,
      };
      asking.holds();
      asking
    }

    /// Whether it has the lock, having taken it if its turn has come.
    pub(crate) fn holds(&mut self) -> bool {
      if let Some(waiting) = &mut self.waiting
        && let Poll::Ready(hold) = waiting.as_mut().poll(&mut Context::from_waker(Waker::noop()))
      {
        self.holding = Some(hold);
        self.waiting = None;
      }
      self.holding.is_some()
    }
  }
}

#[cfg(test)]
mod tests {
  use super::testing::Asking;
  use super::*;

  #[test]
  fn requests_for_one_record_take_turns_in_the_order_they_came() {
    let locks = Locks::default();
    let mut first = Asking::new(&locks, "p1");
    let mut second = Asking::new(&locks, "p1");
    let mut third = Asking::new(&locks, "p1");
    assert!(first.holds());
    assert!(!second.holds() && !third.holds(), "one at a time");

    drop(first);
    assert!(!third.holds(), "not before the one that came before it");
    assert!(second.holds());
    assert!(!third.holds());
    drop(second);
    assert!(third.holds());
  }

  /// The point of it: however many records are held, a request for another one does not wait.
  #[test]
  fn a_request_never_waits_for_a_request_for_another_record() {
    let locks = Locks::default();
    let ids: Vec<String> = (0..5_000).map(|n| format!("p{n}")).collect();
    let mut held = Vec::new();
    for id in &ids {
      let mut asking = Asking::new(&locks, id);
      assert!(asking.holds(), "`{id}` waited with {} other records held", held.len());
      held.push(asking);
    }
    assert_eq!(locks.in_use(), 5_000);
  }

  /// A record has a lock while somebody is at it and not afterwards, so the table does not grow with the catalog.
  #[test]
  fn a_record_is_forgotten_when_the_last_request_for_it_has_left() {
    let locks = Locks::default();
    assert_eq!(locks.in_use(), 0);

    let first = Asking::new(&locks, "p1");
    let second = Asking::new(&locks, "p1");
    let other = Asking::new(&locks, "p2");
    assert_eq!(locks.in_use(), 2, "one for each record, not for each request");

    drop(first);
    assert_eq!(locks.in_use(), 2, "`p1` is still wanted");
    drop(second);
    assert_eq!(locks.in_use(), 1);
    drop(other);
    assert_eq!(locks.in_use(), 0);

    // And it starts again from nothing: the record's next request finds it free.
    assert!(Asking::new(&locks, "p1").holds());
    assert_eq!(locks.in_use(), 0);
  }

  /// A caller that hangs up while its request waits for a record: the request is dropped, and it is as if it
  /// had never asked. Those behind it move up, and nothing of it stays in the table.
  #[test]
  fn a_request_that_gives_up_waiting_holds_up_nobody_and_leaves_nothing() {
    let locks = Locks::default();
    let holder = Asking::new(&locks, "p1");
    let gave_up = Asking::new(&locks, "p1");
    let mut patient = Asking::new(&locks, "p1");

    drop(gave_up);
    assert!(!patient.holds(), "the holder is still there");
    drop(holder);
    assert!(patient.holds(), "the turn passes over the one that left");
    drop(patient);
    assert_eq!(locks.in_use(), 0);

    // The one that leaves last may be one that only ever waited.
    let holder = Asking::new(&locks, "p1");
    let waiting = Asking::new(&locks, "p1");
    drop(holder);
    assert_eq!(locks.in_use(), 1, "its turn has come, whether or not it has looked");
    drop(waiting);
    assert_eq!(locks.in_use(), 0);
  }

  /// A record's id is whatever the caller says it is, and two ids are one record only if they are equal.
  #[test]
  fn ids_are_compared_whole() {
    let locks = Locks::default();
    let _empty = Asking::new(&locks, "");
    let _held = Asking::new(&locks, "p1");
    for other in ["p", "p10", "P1", "p1 ", " p1", "p\u{0031}\u{0301}"] {
      assert!(Asking::new(&locks, other).holds(), "`{other}` is not `p1`");
    }
    assert!(!Asking::new(&locks, "p1").holds());
    assert!(!Asking::new(&locks, "").holds());
  }
}
