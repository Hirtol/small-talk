use eyre::Context;
use sea_orm::prelude::async_trait::async_trait;
use sea_orm::{
    AccessMode, ConnectionTrait, DatabaseConnection, DatabaseTransaction, DbBackend, DbErr, ExecResult, IsolationLevel,
    QueryResult, SqlxSqliteConnector, Statement, StreamTrait, TransactionError, TransactionTrait,
};
use sqlx::sqlite::{SqliteConnectOptions, SqlitePoolOptions};
use sqlx::{Executor, SqlitePool};
use std::error::Error;
use std::future::Future;
use std::ops::Deref;
use std::pin::Pin;

#[derive(Debug, Clone)]
pub struct DatabasePool {
    writer_pool: WriterPool,
    reader_pool: ReaderPool,
}

impl DatabasePool {
    /// Create a new SQLite backed pool.
    ///
    /// `max_writers` is recommended to be `1`.
    pub async fn new_sqlite(
        options: SqliteConnectOptions,
        max_writers: u32,
        max_readers: u32,
    ) -> eyre::Result<DatabasePool> {
        let writer_pool = SqlitePoolOptions::new()
            .max_connections(max_writers)
            .min_connections(max_writers)
            .test_before_acquire(false)
            .connect_with(options.clone())
            .await
            .context("Error in writer pool creation")?;

        let reader_pool = SqlitePoolOptions::new()
            .after_connect(|conn, _| {
                Box::pin(async move {
                    // Runtime enforcement of the read-only property.
                    conn.execute("PRAGMA query_only = true").await?;

                    Ok(())
                })
            })
            .test_before_acquire(false)
            .max_connections(max_readers)
            .min_connections(max_readers)
            .connect_with(options)
            .await
            .context("Error in reader pool creation")?;

        let writer_con = SqlxSqliteConnector::from_sqlx_sqlite_pool(writer_pool);
        let reader_con = SqlxSqliteConnector::from_sqlx_sqlite_pool(reader_pool);

        Ok(Self {
            writer_pool: WriterPool(writer_con),
            reader_pool: ReaderPool(reader_con),
        })
    }

    /// A reader pool.
    ///
    /// The reader should be able to concurrently operate with a writer.
    #[inline]
    pub fn reader(&self) -> &ReaderPool {
        &self.reader_pool
    }

    /// A writer pool.
    ///
    /// For SQLite this would ideally be restricted to a single connection to prevent database locked errors.
    #[inline]
    pub fn writer(&self) -> &WriterPool {
        &self.writer_pool
    }

    pub fn get_sqlx_sqlite_writer(&self) -> &SqlitePool {
        self.writer_pool.0.get_sqlite_connection_pool()
    }
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct WriterPool(DatabaseConnection);

#[repr(transparent)]
#[derive(Debug)]
pub struct WriteTransaction(DatabaseTransaction);

impl WriterPool {
    /// Begin a write-transaction (capable of *both* read, and write).
    #[inline(always)]
    pub async fn begin(&self) -> Result<WriteTransaction, DbErr> {
        Ok(WriteTransaction(self.0.begin().await?))
    }
}

impl WriteTransaction {
    #[inline(always)]
    pub async fn commit(self) -> Result<(), DbErr> {
        self.0.commit().await
    }
}

impl Deref for WriterPool {
    type Target = DatabaseConnection;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for WriteTransaction {
    type Target = DatabaseTransaction;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct ReaderPool(DatabaseConnection);

#[repr(transparent)]
#[derive(Debug)]
pub struct ReadTransaction(DatabaseTransaction);

impl ReaderPool {
    /// Begin a read-only transaction.
    #[inline(always)]
    pub async fn begin(&self) -> Result<ReadTransaction, DbErr> {
        Ok(ReadTransaction(self.0.begin().await?))
    }
}

impl ReadTransaction {
    #[inline(always)]
    pub async fn commit(self) -> Result<(), DbErr> {
        self.0.commit().await
    }
}

impl Deref for ReaderPool {
    type Target = DatabaseConnection;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Deref for ReadTransaction {
    type Target = DatabaseTransaction;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub trait ReadConnection: ConnectionTrait + TransactionTrait + StreamTrait {}

pub trait WriteConnection: ReadConnection {}

macro_rules! impl_stream {
    ($($name:path, $stream:path;)*) => {
        $(
            #[async_trait]
            impl StreamTrait for $name {
                type Stream<'a> = $stream;

                #[inline(always)]
                fn stream<'a>(
                    &'a self,
                    stmt: Statement,
                ) -> Pin<Box<dyn Future<Output = Result<Self::Stream<'a>, DbErr>> + 'a + Send>> {
                    self.0.stream(stmt)
                }
            }
        )*
    };
}

impl_stream!(
    ReadTransaction, sea_orm::TransactionStream<'a>;
    ReaderPool, sea_orm::QueryStream;
    WriteTransaction, sea_orm::TransactionStream<'a>;
    WriterPool, sea_orm::QueryStream;
);

macro_rules! impl_traits {
    ($($name:path, $($traits:path),+;)*) => {
        $(
            #[async_trait]
            impl ConnectionTrait for $name {
                #[inline(always)]
                fn get_database_backend(&self) -> DbBackend {
                    self.0.get_database_backend()
                }

                #[inline(always)]
                async fn execute(&self, stmt: Statement) -> Result<ExecResult, DbErr> {
                    self.0.execute(stmt).await
                }

                #[inline(always)]
                async fn execute_unprepared(&self, sql: &str) -> Result<ExecResult, DbErr> {
                    self.0.execute_unprepared(sql).await
                }

                #[inline(always)]
                async fn query_one(&self, stmt: Statement) -> Result<Option<QueryResult>, DbErr> {
                    self.0.query_one(stmt).await
                }

                #[inline(always)]
                async fn query_all(&self, stmt: Statement) -> Result<Vec<QueryResult>, DbErr> {
                    self.0.query_all(stmt).await
                }
            }

            #[async_trait]
            impl TransactionTrait for $name {
                #[inline(always)]
                async fn begin(&self) -> Result<DatabaseTransaction, DbErr> {
                    self.0.begin().await
                }

                #[inline(always)]
                async fn begin_with_config(&self, isolation_level: Option<IsolationLevel>, access_mode: Option<AccessMode>) -> Result<DatabaseTransaction, DbErr> {
                    self.0.begin_with_config(isolation_level, access_mode).await
                }

                #[inline(always)]
                async fn transaction<F, T, E>(&self, callback: F) -> Result<T, TransactionError<E>> where F: for<'c> FnOnce(&'c DatabaseTransaction) -> Pin<Box<dyn Future<Output=Result<T, E>> + Send + 'c>> + Send, T: Send, E: Error + Send {
                    self.0.transaction(callback).await
                }

                #[inline(always)]
                async fn transaction_with_config<F, T, E>(&self, callback: F, isolation_level: Option<IsolationLevel>, access_mode: Option<AccessMode>) -> Result<T, TransactionError<E>> where F: for<'c> FnOnce(&'c DatabaseTransaction) -> Pin<Box<dyn Future<Output=Result<T, E>> + Send + 'c>> + Send, T: Send, E: Error + Send {
                    self.0.transaction_with_config(callback, isolation_level, access_mode).await
                }
            }

            $(
                impl $traits for $name {}
            )*
        )*
    };
}

impl_traits!(
    WriterPool, WriteConnection, ReadConnection;
    ReaderPool, ReadConnection;
    WriteTransaction, WriteConnection, ReadConnection;
    ReadTransaction, ReadConnection;
);

impl WriteConnection for DatabaseTransaction {}
impl ReadConnection for DatabaseTransaction {}
