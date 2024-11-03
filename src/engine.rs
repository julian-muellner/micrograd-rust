use std::{
    cell::{Ref, RefCell},
    fmt::{self, Debug, Formatter}, 
    hash::{Hash, Hasher}, 
    ops::{Add, AddAssign, Deref, Mul}, 
    rc::Rc
};

/* Marker trait to avoid repetition  */
/* Note: This currently requires the Copy trait. Can be relaxed to be clone at the cost of cloning members. */
pub trait ValueTypeTraits: Default + Debug + Mul<Output=Self> + Add<Output=Self> + AddAssign + Copy {}
impl<T> ValueTypeTraits for T where T: Default + Debug + Mul<Output=Self> + Add<Output=Self> + AddAssign + Copy {}

type DefaultValueType = f32;

#[derive(Debug, PartialEq, Eq, Hash)]
enum Operation {
    None,
    Add,
    Multiply,
}

#[derive(Eq, PartialEq, Debug, Clone)]
pub struct Value<T = DefaultValueType>(Rc<RefCell<ValueImpl<T>>>)
    where T : ValueTypeTraits;

impl<T> Value<T>
    where T: ValueTypeTraits 
{
    fn new(v: ValueImpl<T>) -> Self {
        Value(Rc::new(RefCell::new(v)))
    }

    pub fn from<U>(value: U) -> Value<T> 
    where
        U: Into<T> 
    {
        Value::new(ValueImpl::new(value.into(), Vec::new(), Operation::None))
    }

    pub fn data(&self) -> T {
        self.borrow().data
    }

    pub fn grad(&self) -> T {
        self.borrow().grad
    }
}

impl<T> Default for Value<T>
    where T: ValueTypeTraits 
{
    fn default() -> Self {
        Value::new(ValueImpl::new(T::default(), Vec::new(), Operation::None))
    }
}

impl<T> Hash for Value<T>
    where T: ValueTypeTraits 
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.borrow().hash(state);
    }
}

impl<T> Deref for Value<T> 
    where T: ValueTypeTraits 
{
    type Target = Rc<RefCell<ValueImpl<T>>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, 'b, T> Add<&'b Value<T>> for &'a Value<T>
    where T: ValueTypeTraits
{
    type Output = Value<T>;

    fn add(self, other: &'b Value<T>) -> Self::Output {
        let mut out: ValueImpl<T> = ValueImpl::new(
            self.borrow().data + other.borrow().data, 
            vec![self.clone(), other.clone()], 
            Operation::Add
        );
        
        out.backward = Some(|out_ref| {
            let mut child1 = out_ref.prev[0].borrow_mut();
            let mut child2 = out_ref.prev[1].borrow_mut();

            child1.grad += out_ref.grad;
            child2.grad += out_ref.grad;
        });

        Value::new(out)
    }
}

/* Consuming add, convenience method */
impl<T> Add for Value<T> 
    where T: ValueTypeTraits 
{
    type Output = Value<T>;
    fn add(self, rhs: Self) -> Self::Output {
        &self + &rhs
    }
}

impl<'a, 'b, T> Mul<&'b Value<T>> for &'a Value<T> 
    where T: ValueTypeTraits
{
    type Output = Value<T>;

    fn mul(self, other: &'b Value<T>) -> Self::Output {
        let mut out: ValueImpl<T> = ValueImpl::new(
            self.borrow().data * other.borrow().data, 
            vec![self.clone(), other.clone()], 
            Operation::Multiply
        );
        
        out.backward = Some(|out_ref| {
            let mut child1 = out_ref.prev[0].borrow_mut();
            let mut child2 = out_ref.prev[1].borrow_mut();

            child1.grad += child2.data * out_ref.grad;
            child2.grad += child1.data * out_ref.grad;
        });

        Value::new(out)
    }
}

/* Consuming add, convenience method */
impl<T> Mul for Value<T> 
    where T: ValueTypeTraits 
{
    type Output = Value<T>;
    fn mul(self, rhs: Self) -> Self::Output {
        &self * &rhs
    }
}

type BackpropFn<T> = Option<fn(value: Ref<ValueImpl<T>>)>;

pub struct ValueImpl<T = DefaultValueType>
    where T: ValueTypeTraits
{
    data: T,
    grad: T,
    backward: BackpropFn<T>,
    prev: Vec<Value<T>>,
    op: Operation,
}

impl<T> ValueImpl<T>
    where T: ValueTypeTraits
{
    fn new(data: T, children: Vec<Value<T>>, op: Operation) -> Self {
        ValueImpl {
            data: data, 
            grad: T::default(), 
            backward: None, 
            prev: children,
            op,
        }
    }
}

impl<T> Debug for ValueImpl<T> 
    where T: ValueTypeTraits {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ValueImpl")
            .field("data", &self.data)
            .field("grad", &self.grad)
            .field("prev", &self.prev)
            .field("op", &self.op)
            .finish()
    }
}

/* Implement Equality, PartialEquality and Hash for References to ValueImpl. Only the same object is considered to be identical.  */
impl<T> PartialEq for ValueImpl<T>
    where T: ValueTypeTraits 
{
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl<T> Eq for ValueImpl<T>
    where T: ValueTypeTraits {}

impl<T> Hash for ValueImpl<T> 
    where T: ValueTypeTraits 
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::ptr::hash(self, state)
    }
}


/* Tests */

#[cfg(test)]
mod tests {
    // Note this useful idiom: importing names from outer (for mod tests) scope.
    use super::*;

    #[test]
    fn test_add_float() {
        let v1 = Value::from(5.1);
        let v2 = Value::from(-1.1);

        let two = Value::from(2.0);
        let three = Value::from(3.0);
        assert_eq!((&v1 + &v2).data(), 4.0);

        let v3 : Value = &v1 + &v2;
        assert_eq!(v3.data(), 4.0);

        assert_eq!((&v1 + &two).data(), 7.1);
        assert_eq!((&three + &v2).data(), 1.9);

        let v4 = v1 + v2; // consuming add
        assert_eq!(v4.data(), 4.0);
    }

    #[test]
    fn test_add_int() {
        let v1: Value<i32> = Value::from(5);
        let v2 = Value::from(-1);

        assert_eq!((&v1 + &v2).data(), 4);

        let v4 = v1 + v2; // consuming add
        assert_eq!(v4.data(), 4);
    }

    #[derive(Debug, Copy, Clone)]
    struct Wrap(i32);
    impl Add for Wrap {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            Wrap(self.0 + rhs.0)
        }
    }
    impl Mul for Wrap {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self::Output {
            Wrap(self.0 * rhs.0)
        }
    }
    impl AddAssign for Wrap {
        fn add_assign(&mut self, rhs: Self) {
            self.0 += rhs.0
        }
    }
    impl Default for Wrap {
        fn default() -> Self {
            Wrap(0)
        }
    }

    #[test]
    fn test_add_struct() {
        let v1: Value<Wrap> = Value::from(Wrap(5));
        let v2 = Value::from(Wrap(-1));

        assert_eq!((&v1 + &v2).data().0, 4);

        let v4 = v1 + v2; // consuming add
        assert_eq!(v4.data().0, 4);
    }
}