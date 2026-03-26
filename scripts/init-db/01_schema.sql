-- =============================================================================
-- ERP Agentic Chatbot — Database Schema
-- Chạy tự động khi PostgreSQL container khởi động lần đầu
-- =============================================================================

-- ── Extensions ────────────────────────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- full-text search hỗ trợ

-- =============================================================================
-- SYSTEM TABLES
-- =============================================================================

-- ── API Keys (auth) ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS api_keys (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id   VARCHAR(100) NOT NULL,
    user_id     VARCHAR(100) NOT NULL,
    key_hash    VARCHAR(64)  NOT NULL UNIQUE,   -- SHA-256 của API key thực
    name        VARCHAR(200),                   -- nhãn mô tả key
    active      BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    expires_at  TIMESTAMPTZ,
    last_used_at TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_api_keys_hash     ON api_keys (key_hash);
CREATE INDEX IF NOT EXISTS idx_api_keys_tenant   ON api_keys (tenant_id);

-- ── Documents (upload tracking) ───────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id            UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id     VARCHAR(100) NOT NULL,
    filename      VARCHAR(500) NOT NULL,
    file_size     BIGINT,
    uploaded_by   VARCHAR(100),
    status        VARCHAR(20)  NOT NULL DEFAULT 'processing'
                  CHECK (status IN ('processing','indexed','failed','deleted')),
    chunk_count   INT          DEFAULT 0,
    error_message TEXT,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    indexed_at    TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_tenant_file
    ON documents (tenant_id, filename) WHERE status != 'deleted';
CREATE INDEX IF NOT EXISTS idx_documents_tenant ON documents (tenant_id);

-- =============================================================================
-- ERP BUSINESS TABLES
-- =============================================================================

-- ── Tenants ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tenants (
    id          VARCHAR(100) PRIMARY KEY,
    name        VARCHAR(200) NOT NULL,
    active      BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- ── Departments ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS departments (
    id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id   VARCHAR(100) NOT NULL,
    name        VARCHAR(200) NOT NULL,
    code        VARCHAR(50),
    manager_id  UUID,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_departments_tenant ON departments (tenant_id);

-- ── Employees ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS employees (
    id            UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id     VARCHAR(100) NOT NULL,
    code          VARCHAR(50),
    full_name     VARCHAR(200) NOT NULL,
    email         VARCHAR(200),
    phone         VARCHAR(20),
    department_id UUID        REFERENCES departments(id),
    position      VARCHAR(100),
    salary        NUMERIC(15,2),
    hire_date     DATE,
    status        VARCHAR(20)  DEFAULT 'active'
                  CHECK (status IN ('active','inactive','terminated')),
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_employees_tenant ON employees (tenant_id);
CREATE INDEX IF NOT EXISTS idx_employees_dept   ON employees (department_id);

-- ── Suppliers ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS suppliers (
    id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id   VARCHAR(100) NOT NULL,
    code        VARCHAR(50),
    name        VARCHAR(200) NOT NULL,
    email       VARCHAR(200),
    phone       VARCHAR(20),
    address     TEXT,
    tax_code    VARCHAR(50),
    active      BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_suppliers_tenant ON suppliers (tenant_id);

-- ── Customers ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS customers (
    id          UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id   VARCHAR(100) NOT NULL,
    code        VARCHAR(50),
    name        VARCHAR(200) NOT NULL,
    email       VARCHAR(200),
    phone       VARCHAR(20),
    address     TEXT,
    tax_code    VARCHAR(50),
    type        VARCHAR(20)  DEFAULT 'individual'
                CHECK (type IN ('individual','company')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_customers_tenant ON customers (tenant_id);

-- ── Products ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS products (
    id           UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id    VARCHAR(100) NOT NULL,
    code         VARCHAR(100) NOT NULL,
    name         VARCHAR(300) NOT NULL,
    unit         VARCHAR(50),
    category     VARCHAR(100),
    cost_price   NUMERIC(15,2) DEFAULT 0,
    sell_price   NUMERIC(15,2) DEFAULT 0,
    active       BOOLEAN      NOT NULL DEFAULT TRUE,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_products_code ON products (tenant_id, code);
CREATE INDEX IF NOT EXISTS idx_products_tenant ON products (tenant_id);

-- ── Inventory ─────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS inventory (
    id           UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id    VARCHAR(100) NOT NULL,
    product_id   UUID        NOT NULL REFERENCES products(id),
    warehouse    VARCHAR(100) DEFAULT 'main',
    quantity     NUMERIC(15,3) NOT NULL DEFAULT 0,
    min_quantity NUMERIC(15,3) DEFAULT 0,
    updated_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_inventory_product_wh
    ON inventory (tenant_id, product_id, warehouse);
CREATE INDEX IF NOT EXISTS idx_inventory_tenant ON inventory (tenant_id);

-- ── Orders ────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS orders (
    id           UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id    VARCHAR(100) NOT NULL,
    order_number VARCHAR(50),
    customer_id  UUID        REFERENCES customers(id),
    employee_id  UUID        REFERENCES employees(id),
    status       VARCHAR(20)  NOT NULL DEFAULT 'draft'
                 CHECK (status IN ('draft','confirmed','shipped','delivered','cancelled')),
    total_amount NUMERIC(15,2) DEFAULT 0,
    discount     NUMERIC(15,2) DEFAULT 0,
    tax          NUMERIC(15,2) DEFAULT 0,
    notes        TEXT,
    order_date   DATE         NOT NULL DEFAULT CURRENT_DATE,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_orders_tenant      ON orders (tenant_id);
CREATE INDEX IF NOT EXISTS idx_orders_customer    ON orders (customer_id);
CREATE INDEX IF NOT EXISTS idx_orders_date        ON orders (tenant_id, order_date DESC);

-- ── Order Items ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS order_items (
    id           UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id    VARCHAR(100) NOT NULL,
    order_id     UUID        NOT NULL REFERENCES orders(id) ON DELETE CASCADE,
    product_id   UUID        NOT NULL REFERENCES products(id),
    quantity     NUMERIC(15,3) NOT NULL,
    unit_price   NUMERIC(15,2) NOT NULL,
    discount     NUMERIC(15,2) DEFAULT 0,
    line_total   NUMERIC(15,2) GENERATED ALWAYS AS
                 (quantity * unit_price - discount) STORED
);
CREATE INDEX IF NOT EXISTS idx_order_items_order  ON order_items (order_id);
CREATE INDEX IF NOT EXISTS idx_order_items_tenant ON order_items (tenant_id);

-- ── Purchase Orders ───────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS purchase_orders (
    id           UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id    VARCHAR(100) NOT NULL,
    po_number    VARCHAR(50),
    supplier_id  UUID        REFERENCES suppliers(id),
    status       VARCHAR(20)  NOT NULL DEFAULT 'draft'
                 CHECK (status IN ('draft','sent','received','cancelled')),
    total_amount NUMERIC(15,2) DEFAULT 0,
    order_date   DATE         NOT NULL DEFAULT CURRENT_DATE,
    expected_date DATE,
    created_at   TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_po_tenant ON purchase_orders (tenant_id);
CREATE INDEX IF NOT EXISTS idx_po_date   ON purchase_orders (tenant_id, order_date DESC);

-- ── Invoices ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS invoices (
    id             UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id      VARCHAR(100) NOT NULL,
    invoice_number VARCHAR(50),
    order_id       UUID        REFERENCES orders(id),
    customer_id    UUID        REFERENCES customers(id),
    status         VARCHAR(20)  NOT NULL DEFAULT 'unpaid'
                   CHECK (status IN ('unpaid','partial','paid','overdue','cancelled')),
    subtotal       NUMERIC(15,2) DEFAULT 0,
    tax            NUMERIC(15,2) DEFAULT 0,
    total          NUMERIC(15,2) DEFAULT 0,
    paid_amount    NUMERIC(15,2) DEFAULT 0,
    due_date       DATE,
    issued_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_invoices_tenant   ON invoices (tenant_id);
CREATE INDEX IF NOT EXISTS idx_invoices_customer ON invoices (customer_id);
CREATE INDEX IF NOT EXISTS idx_invoices_status   ON invoices (tenant_id, status);

-- ── Payments ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS payments (
    id             UUID        PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id      VARCHAR(100) NOT NULL,
    invoice_id     UUID        REFERENCES invoices(id),
    amount         NUMERIC(15,2) NOT NULL,
    method         VARCHAR(50)   DEFAULT 'cash'
                   CHECK (method IN ('cash','bank_transfer','card','other')),
    reference      VARCHAR(200),
    paid_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    created_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_payments_tenant  ON payments (tenant_id);
CREATE INDEX IF NOT EXISTS idx_payments_invoice ON payments (invoice_id);
CREATE INDEX IF NOT EXISTS idx_payments_date    ON payments (tenant_id, paid_at DESC);
