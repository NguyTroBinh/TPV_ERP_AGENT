-- =============================================================================
-- Seed data — tenant mặc định + dữ liệu mẫu để test
-- =============================================================================

-- ── Tenant mặc định ───────────────────────────────────────────────────────────
INSERT INTO tenants (id, name) VALUES ('tpv', 'Giải pháp Công nghệ và chuyển đổi xanh')
ON CONFLICT (id) DO NOTHING;

-- ── Departments ───────────────────────────────────────────────────────────────
INSERT INTO departments (id, tenant_id, name, code) VALUES
    ('00000000-0000-0000-0000-000000000001', 'tpv', 'Kinh doanh',      'KD'),
    ('00000000-0000-0000-0000-000000000002', 'tpv', 'Kế toán',         'KT'),
    ('00000000-0000-0000-0000-000000000003', 'tpv', 'Kho vận',         'KV'),
    ('00000000-0000-0000-0000-000000000004', 'tpv', 'Nhân sự',         'NS'),
    ('00000000-0000-0000-0000-000000000005', 'tpv', 'Công nghệ thông tin', 'IT')
ON CONFLICT DO NOTHING;

-- ── Employees ─────────────────────────────────────────────────────────────────
INSERT INTO employees (tenant_id, code, full_name, email, department_id, position, salary, hire_date) VALUES
    ('tpv', 'NV001', 'Nguyễn Văn An',   'an.nv@demo.com',    '00000000-0000-0000-0000-000000000001', 'Trưởng phòng KD',  25000000, '2020-01-15'),
    ('tpv', 'NV002', 'Trần Thị Bình',   'binh.tt@demo.com',  '00000000-0000-0000-0000-000000000001', 'Nhân viên KD',     15000000, '2021-03-01'),
    ('tpv', 'NV003', 'Lê Văn Cường',    'cuong.lv@demo.com', '00000000-0000-0000-0000-000000000002', 'Kế toán trưởng',   20000000, '2019-06-10'),
    ('tpv', 'NV004', 'Phạm Thị Dung',   'dung.pt@demo.com',  '00000000-0000-0000-0000-000000000003', 'Thủ kho',          12000000, '2022-01-20'),
    ('tpv', 'NV005', 'Hoàng Văn Em',    'em.hv@demo.com',    '00000000-0000-0000-0000-000000000005', 'Lập trình viên',   22000000, '2021-07-01')
ON CONFLICT DO NOTHING;

-- ── Customers ─────────────────────────────────────────────────────────────────
INSERT INTO customers (tenant_id, code, name, email, phone, type) VALUES
    ('tpv', 'KH001', 'Công ty TNHH ABC',          'mua@abc.com',    '0901234567', 'company'),
    ('tpv', 'KH002', 'Công ty Cổ phần XYZ',       'order@xyz.vn',   '0912345678', 'company'),
    ('tpv', 'KH003', 'Nguyễn Thị Hoa',            'hoa@gmail.com',  '0923456789', 'individual'),
    ('tpv', 'KH004', 'Tập đoàn Phú Quý',          'info@phuquy.vn', '0834567890', 'company'),
    ('tpv', 'KH005', 'Trần Văn Giang',             'giang@yahoo.com','0845678901', 'individual')
ON CONFLICT DO NOTHING;

-- ── Suppliers ─────────────────────────────────────────────────────────────────
INSERT INTO suppliers (tenant_id, code, name, email, phone) VALUES
    ('tpv', 'NCC001', 'Nhà cung cấp Miền Bắc', 'contact@mienbac.vn', '0956789012'),
    ('tpv', 'NCC002', 'Công ty NK Sài Gòn',    'import@saigon.vn',   '0867890123'),
    ('tpv', 'NCC003', 'Xưởng sản xuất Đồng Nai','sx@dongnai.vn',     '0278901234')
ON CONFLICT DO NOTHING;

-- ── Products ──────────────────────────────────────────────────────────────────
INSERT INTO products (tenant_id, code, name, unit, category, cost_price, sell_price) VALUES
    ('tpv', 'SP001', 'Laptop Dell Inspiron 15',   'Cái',  'Điện tử',    18000000, 22000000),
    ('tpv', 'SP002', 'Chuột không dây Logitech',  'Cái',  'Phụ kiện',     250000,   350000),
    ('tpv', 'SP003', 'Bàn phím cơ Keychron K2',   'Cái',  'Phụ kiện',    1500000,  2100000),
    ('tpv', 'SP004', 'Màn hình LG 27 inch 4K',    'Cái',  'Màn hình',    7000000,  9500000),
    ('tpv', 'SP005', 'Tai nghe Sony WH-1000XM5',  'Cái',  'Âm thanh',    6500000,  8900000),
    ('tpv', 'SP006', 'Ổ cứng SSD 1TB Samsung',    'Cái',  'Lưu trữ',     2200000,  2900000),
    ('tpv', 'SP007', 'RAM DDR5 16GB Kingston',     'Thanh','Linh kiện',    900000,  1200000),
    ('tpv', 'SP008', 'Balo laptop Samsonite',      'Cái',  'Phụ kiện',     800000,  1100000)
ON CONFLICT DO NOTHING;

-- ── Inventory ─────────────────────────────────────────────────────────────────
INSERT INTO inventory (tenant_id, product_id, warehouse, quantity, min_quantity)
SELECT 'tpv', id, 'Kho chính',
    CASE code
        WHEN 'SP001' THEN 50
        WHEN 'SP002' THEN 200
        WHEN 'SP003' THEN 80
        WHEN 'SP004' THEN 30
        WHEN 'SP005' THEN 25
        WHEN 'SP006' THEN 120
        WHEN 'SP007' THEN 150
        WHEN 'SP008' THEN 90
    END,
    CASE code
        WHEN 'SP001' THEN 10
        WHEN 'SP002' THEN 30
        ELSE 5
    END
FROM products WHERE tenant_id = 'tpv'
ON CONFLICT DO NOTHING;

-- ── Orders (3 tháng gần nhất) ─────────────────────────────────────────────────
DO $$
DECLARE
    v_cust1 UUID; v_cust2 UUID; v_cust3 UUID;
    v_emp1  UUID;
    v_ord1  UUID; v_ord2  UUID; v_ord3  UUID;
    v_sp001 UUID; v_sp002 UUID; v_sp003 UUID; v_sp004 UUID;
BEGIN
    SELECT id INTO v_cust1 FROM customers WHERE tenant_id='tpv' AND code='KH001';
    SELECT id INTO v_cust2 FROM customers WHERE tenant_id='tpv' AND code='KH002';
    SELECT id INTO v_cust3 FROM customers WHERE tenant_id='tpv' AND code='KH003';
    SELECT id INTO v_emp1  FROM employees  WHERE tenant_id='tpv' AND code='NV001';
    SELECT id INTO v_sp001 FROM products   WHERE tenant_id='tpv' AND code='SP001';
    SELECT id INTO v_sp002 FROM products   WHERE tenant_id='tpv' AND code='SP002';
    SELECT id INTO v_sp003 FROM products   WHERE tenant_id='tpv' AND code='SP003';
    SELECT id INTO v_sp004 FROM products   WHERE tenant_id='tpv' AND code='SP004';

    -- Đơn hàng 1
    INSERT INTO orders (id, tenant_id, order_number, customer_id, employee_id, status, total_amount, order_date)
    VALUES (uuid_generate_v4(), 'tpv', 'DH-2026-001', v_cust1, v_emp1, 'delivered', 46350000, CURRENT_DATE - 45)
    ON CONFLICT DO NOTHING
    RETURNING id INTO v_ord1;

    IF v_ord1 IS NOT NULL THEN
        INSERT INTO order_items (tenant_id, order_id, product_id, quantity, unit_price) VALUES
            ('tpv', v_ord1, v_sp001, 2, 22000000),
            ('tpv', v_ord1, v_sp002, 3,   350000);
    END IF;

    -- Đơn hàng 2
    INSERT INTO orders (id, tenant_id, order_number, customer_id, employee_id, status, total_amount, order_date)
    VALUES (uuid_generate_v4(), 'tpv', 'DH-2026-002', v_cust2, v_emp1, 'shipped', 11700000, CURRENT_DATE - 10)
    ON CONFLICT DO NOTHING
    RETURNING id INTO v_ord2;

    IF v_ord2 IS NOT NULL THEN
        INSERT INTO order_items (tenant_id, order_id, product_id, quantity, unit_price) VALUES
            ('tpv', v_ord2, v_sp003, 3, 2100000),
            ('tpv', v_ord2, v_sp002, 15,  350000);
    END IF;

    -- Đơn hàng 3
    INSERT INTO orders (id, tenant_id, order_number, customer_id, employee_id, status, total_amount, order_date)
    VALUES (uuid_generate_v4(), 'tpv', 'DH-2026-003', v_cust3, v_emp1, 'confirmed', 9500000, CURRENT_DATE - 2)
    ON CONFLICT DO NOTHING
    RETURNING id INTO v_ord3;

    IF v_ord3 IS NOT NULL THEN
        INSERT INTO order_items (tenant_id, order_id, product_id, quantity, unit_price) VALUES
            ('tpv', v_ord3, v_sp004, 1, 9500000);
    END IF;
END $$;
