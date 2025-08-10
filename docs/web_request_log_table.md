# Web Request Log Table

The `WebRequestLog` table records outbound HTTP requests made by EnkiBot for auditing and diagnostics.

```sql
CREATE TABLE WebRequestLog (
  LogID INT IDENTITY(1,1) PRIMARY KEY,
  Url NVARCHAR(1024) NOT NULL,
  Method NVARCHAR(16) NOT NULL,
  StatusCode INT NULL,
  DurationMs INT NULL,
  Error NVARCHAR(512) NULL,
  Timestamp DATETIME2 DEFAULT GETDATE() NOT NULL
);
```
